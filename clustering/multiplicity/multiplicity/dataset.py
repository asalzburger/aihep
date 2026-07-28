"""Turns one or more sensor-shaped run directories into a fixed-size
pixel-matrix classification dataset: one (matrix, n_particles) example per
*real* cluster -- a cluster with at least one contributing truth particle,
i.e. excluding pure-noise clusters, which aren't a 1/2/3-particle example
at all (see `sensor`'s README on readout thresholds/noise for why those
exist in the first place).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import Format, read_run


@dataclass
class Dataset:
    matrices: np.ndarray
    """(n_examples, n_x, n_y) float32 -- charge per pixel, cluster-local
    coordinates centered within the fixed (n_x, n_y) canvas."""
    n_particles: np.ndarray
    """(n_examples,) int -- ground-truth particle count per cluster."""
    event_id: np.ndarray
    cluster_id: np.ndarray
    source: np.ndarray
    """Which input directory each example came from (provenance)."""
    matrix_shape: tuple[int, int]


def _label_clusters(hits: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """One row per (event_id, cluster_id) with at least one contributing
    truth particle: n_particles = the number of distinct particle_ids that
    deposited charge somewhere in that cluster."""
    merged = hits[["event_id", "ix", "iy", "cluster_id"]].merge(
        contributions[["event_id", "ix", "iy", "particle_id"]], on=["event_id", "ix", "iy"], how="inner"
    )
    labels = merged.groupby(["event_id", "cluster_id"])["particle_id"].nunique()
    return labels.rename("n_particles").reset_index()


def compute_matrix_shape(hits: pd.DataFrame, clusters: pd.DataFrame, contributions: pd.DataFrame) -> tuple[int, int]:
    """(n_x, n_y): the largest x/y pixel span among *real* clusters (see
    `_label_clusters`), plus 1 -- the fixed matrix size every cluster in
    the dataset gets embedded into, with room to spare even for the
    largest cluster seen."""
    labels = _label_clusters(hits, contributions)
    real = clusters.merge(labels[["event_id", "cluster_id"]], on=["event_id", "cluster_id"])
    if real.empty:
        raise ValueError("no cluster in this data has any contributing truth particle")
    return int(real["x_span_pixels"].max()) + 1, int(real["y_span_pixels"].max()) + 1


def _cluster_matrix(pixels: pd.DataFrame, n_x: int, n_y: int) -> np.ndarray:
    """Embed one cluster's (ix, iy, charge) pixels into an (n_x, n_y)
    matrix, centered: the cluster's own bounding box is centered within
    the fixed canvas rather than pinned to a corner, so the network sees a
    consistent, position-independent representation regardless of cluster
    size."""
    ix_min, iy_min = pixels["ix"].min(), pixels["iy"].min()
    x_span = int(pixels["ix"].max() - ix_min + 1)
    y_span = int(pixels["iy"].max() - iy_min + 1)
    if x_span > n_x or y_span > n_y:
        raise ValueError(f"cluster span ({x_span}, {y_span}) exceeds the fixed matrix shape ({n_x}, {n_y})")

    x_offset = (n_x - x_span) // 2
    y_offset = (n_y - y_span) // 2

    matrix = np.zeros((n_x, n_y), dtype=np.float32)
    local_ix = (pixels["ix"] - ix_min + x_offset).to_numpy()
    local_iy = (pixels["iy"] - iy_min + y_offset).to_numpy()
    matrix[local_ix, local_iy] = pixels["charge"].to_numpy(dtype=np.float32)
    return matrix


def build_dataset(
    input_dirs: list[str | Path], fmt: Format = "arrow", matrix_shape: tuple[int, int] | None = None
) -> Dataset:
    """Build a Dataset from one or more sensor-shaped run directories.

    matrix_shape: fix it to a shape already computed elsewhere (e.g. from a
    training run, so an independent evaluation set uses the exact same
    input size even if its own clusters happen to be smaller); if omitted,
    it's computed from this data (see `compute_matrix_shape`).
    """
    per_dir: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for input_dir in input_dirs:
        hits, clusters, _truth, contributions = read_run(input_dir, fmt)
        per_dir.append((str(input_dir), hits, clusters, contributions))

    if matrix_shape is None:
        all_hits = pd.concat([h for _, h, _, _ in per_dir], ignore_index=True)
        all_clusters = pd.concat([c for _, _, c, _ in per_dir], ignore_index=True)
        all_contributions = pd.concat([k for _, _, _, k in per_dir], ignore_index=True)
        matrix_shape = compute_matrix_shape(all_hits, all_clusters, all_contributions)
    n_x, n_y = matrix_shape

    matrices: list[np.ndarray] = []
    n_particles: list[int] = []
    event_ids: list[int] = []
    cluster_ids: list[int] = []
    sources: list[str] = []

    for source, hits, _clusters, contributions in per_dir:
        labels = _label_clusters(hits, contributions)
        label_lookup = dict(zip(zip(labels["event_id"], labels["cluster_id"]), labels["n_particles"]))
        real_hits = hits.merge(labels[["event_id", "cluster_id"]], on=["event_id", "cluster_id"], how="inner")

        for (event_id, cluster_id), pixels in real_hits.groupby(["event_id", "cluster_id"], sort=False):
            matrices.append(_cluster_matrix(pixels, n_x, n_y))
            n_particles.append(int(label_lookup[(event_id, cluster_id)]))
            event_ids.append(int(event_id))
            cluster_ids.append(int(cluster_id))
            sources.append(source)

    return Dataset(
        matrices=np.stack(matrices) if matrices else np.zeros((0, n_x, n_y), dtype=np.float32),
        n_particles=np.array(n_particles, dtype=np.int64),
        event_id=np.array(event_ids, dtype=np.int64),
        cluster_id=np.array(cluster_ids, dtype=np.int64),
        source=np.array(sources),
        matrix_shape=(n_x, n_y),
    )
