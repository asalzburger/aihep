"""Group digitized hits into clusters: 1D connected components of adjacent
cell indices, per (event_id, layer_id). Deliberately plain Python -- unlike
`sensor`'s 2D pixel-grid case, a single layer's cells are a sorted
1D sequence, so adjacency is just "gap <= connectivity_gap" between
consecutive sorted cells; no `scipy.ndimage.label` needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .edm import CLUSTERED_HITS_COLUMNS, CLUSTERS_COLUMNS


def cluster_hits(hits_df: pd.DataFrame, connectivity_gap: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign a cluster_id to every digitized hit and aggregate cluster-level
    quantities. `cluster_id` is unique within an event (numbered across all
    of that event's layers); hits with no `cell_index` (layer has no pitch)
    get `cluster_id = -1` and are excluded from the returned `clusters`.

    Returns (hits_df_with_cluster_id, clusters_df).
    """
    if hits_df.empty:
        hits_out = hits_df.copy()
        hits_out["cluster_id"] = pd.Series(dtype=int)
        return hits_out[CLUSTERED_HITS_COLUMNS], pd.DataFrame(columns=CLUSTERS_COLUMNS)

    hits_parts: list[pd.DataFrame] = []
    cluster_rows: list[dict] = []

    for event_id, event_hits in hits_df.groupby("event_id", sort=True):
        event_hits = event_hits.copy()
        cluster_id_col = pd.Series(-1, index=event_hits.index, dtype=int)
        next_cluster_id = 0

        for layer_id, layer_hits in event_hits.groupby("layer_id", sort=True):
            digitized = layer_hits.dropna(subset=["cell_index"])
            if digitized.empty:
                continue
            ordered = digitized.sort_values("cell_index")
            cells = ordered["cell_index"].to_numpy()

            group_ids = np.zeros(len(cells), dtype=int)
            for i in range(1, len(cells)):
                gap_starts_new_cluster = cells[i] - cells[i - 1] > connectivity_gap
                group_ids[i] = group_ids[i - 1] + 1 if gap_starts_new_cluster else group_ids[i - 1]

            for g in range(group_ids[-1] + 1):
                idx = ordered.index[group_ids == g]
                cluster_id_col.loc[idx] = next_cluster_id
                rows = event_hits.loc[idx]
                cluster_rows.append(
                    dict(
                        event_id=event_id,
                        layer_id=layer_id,
                        cluster_id=next_cluster_id,
                        n_cells=int(rows["cell_index"].nunique()),
                        n_hits=len(rows),
                        s_centroid=float(rows["s_local"].mean()),
                        x_centroid=float(rows["x"].mean()),
                        y_centroid=float(rows["y"].mean()),
                    )
                )
                next_cluster_id += 1

        event_hits["cluster_id"] = cluster_id_col
        hits_parts.append(event_hits)

    hits_out = pd.concat(hits_parts, ignore_index=True)[CLUSTERED_HITS_COLUMNS]
    clusters_df = pd.DataFrame(cluster_rows, columns=CLUSTERS_COLUMNS)
    return hits_out, clusters_df
