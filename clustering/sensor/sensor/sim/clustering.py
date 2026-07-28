"""Group hit pixels into clusters via connected-components on the pixel grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label

from ..edm import CLUSTERED_HITS_COLUMNS, CLUSTERS_COLUMNS
from .config import DetectorConfig


def cluster_hits(
    hits_df: pd.DataFrame,
    detector: DetectorConfig,
    connectivity: int = 8,
    readout_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign a cluster_id to each hit pixel (connected components per event)
    and aggregate cluster-level quantities.

    readout_threshold: pixels with charge at or below this are dropped
        before clustering, i.e. treated as not read out (mimicking a real
        front-end comparator threshold on top of whatever digitization
        already decided was a "hit"). Default 0.0 is a no-op given hits are
        already produced with charge > 0.

    Each cluster gets two parallel centroid definitions for comparison:
    charge-weighted (x_centroid_um/y_centroid_um) and digital, i.e. the
    unweighted mean of pixel centers (x_centroid_digital_um/
    y_centroid_digital_um).

    Returns (hits_df_with_cluster_id, clusters_df).
    """
    hits_df = hits_df[hits_df["charge"] > readout_threshold]

    if hits_df.empty:
        hits_out = hits_df.copy()
        hits_out["cluster_id"] = pd.Series(dtype=int)
        return hits_out[CLUSTERED_HITS_COLUMNS], pd.DataFrame(columns=CLUSTERS_COLUMNS)

    structure = generate_binary_structure(2, 2 if connectivity == 8 else 1)

    hits_parts: list[pd.DataFrame] = []
    cluster_rows: list[dict] = []
    for event_id, event_hits in hits_df.groupby("event_id", sort=True):
        mask = np.zeros((detector.n_pixels_x, detector.n_pixels_y), dtype=bool)
        ix, iy = event_hits["ix"].to_numpy(), event_hits["iy"].to_numpy()
        mask[ix, iy] = True
        labeled, _ = label(mask, structure=structure)

        event_hits = event_hits.copy()
        event_hits["cluster_id"] = labeled[ix, iy] - 1
        hits_parts.append(event_hits)

        for cluster_id, pixels in event_hits.groupby("cluster_id"):
            charge_sum = pixels["charge"].sum()
            cluster_rows.append(
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

    hits_out = pd.concat(hits_parts, ignore_index=True)[CLUSTERED_HITS_COLUMNS]
    clusters_df = pd.DataFrame(cluster_rows, columns=CLUSTERS_COLUMNS)
    return hits_out, clusters_df
