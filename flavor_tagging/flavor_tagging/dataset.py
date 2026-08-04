"""Turns a reconstructed run's `tracks`/`clusters` tables (from
`detectorreco2d`) into a fixed-size per-jet feature/label dataset for the
b-tagging classifier: one example per truth jet (`jet_id != -1`).

Every jet gets the same feature layout regardless of its own track count --
a deliberately *overcommitted* fixed-size array of its leading tracks'
impact parameters (`d0`, sorted by descending reconstructed `pt` -- the most
energetic tracks first, zero-padded past however many tracks that jet
actually had; see `compute_n_track_slots`), plus four scalar features:
`n_tracks`, `n_muons`, and a calorimeter-side summary
(`total_cluster_energy`, `n_clusters`).

This is deliberately the same truth (`is_b_jet`) and the same tables
`flavor_tagging.vis`'s validation plots already use -- the classifier is
being asked to recover, from reconstructed quantities alone, exactly the
split those plots show by eye.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pipeline import summarize_jets

#: Scalar (non-d0-slot) features, in the fixed order they're appended in.
SCALAR_FEATURE_NAMES = ("n_tracks", "n_muons", "total_cluster_energy", "n_clusters")


@dataclass
class Dataset:
    features: np.ndarray
    """(n_examples, n_track_slots + len(SCALAR_FEATURE_NAMES)) float32,
    unnormalized -- see `compute_standardization`/`standardize`."""
    is_b_jet: np.ndarray
    """(n_examples,) int64, 0/1 -- the label."""
    event_id: np.ndarray
    jet_id: np.ndarray
    n_track_slots: int
    feature_names: list[str]


def compute_n_track_slots(tracks: pd.DataFrame) -> int:
    """The largest per-jet track count in `tracks` -- the fixed, deliberately
    overcommitted number of leading-track `d0` slots every jet's feature
    vector gets embedded into. Most jets (especially light jets, which run
    fewer tracks on average -- see `ParticleGunConfig.b_jet_track_boost`)
    fill only a fraction of these slots; the rest stay zero-padded."""
    jetted = tracks[tracks["jet_id"] != -1]
    if not len(jetted):
        raise ValueError("no jet-mode tracks in this data (every jet_id is -1)")
    return int(jetted.groupby(["event_id", "jet_id"]).size().max())


def _jet_d0_slots(jet_tracks: pd.DataFrame, n_track_slots: int) -> np.ndarray:
    """This jet's leading (highest reconstructed-`pt`) tracks' `d0`, most
    energetic first, zero-padded to exactly `n_track_slots`. Truncated (kept
    to the `n_track_slots` highest-`pt` tracks) rather than raising if this
    particular jet has more tracks than that -- expected to be rare (an
    independent evaluation set's own busiest jet could exceed a
    `n_track_slots` computed from a different, training run), and dropping
    only the softest excess tracks is a reasonable, information-preserving
    fallback rather than a hard failure.
    """
    ordered = jet_tracks.sort_values("pt", ascending=False)["d0"].to_numpy(dtype=np.float32)
    slots = np.zeros(n_track_slots, dtype=np.float32)
    n = min(len(ordered), n_track_slots)
    slots[:n] = ordered[:n]
    return slots


def build_dataset(tracks: pd.DataFrame, clusters: pd.DataFrame, n_track_slots: int | None = None) -> Dataset:
    """Build a Dataset from one reconstructed run's `tracks`/`clusters`.

    `n_track_slots`: fix it to a value already computed elsewhere (e.g. from
    a training run, so an independent evaluation set uses the exact same
    input size even if its own busiest jet has more or fewer tracks); if
    omitted, it's computed from this data (see `compute_n_track_slots`).
    """
    jetted_tracks = tracks[tracks["jet_id"] != -1]
    if n_track_slots is None:
        n_track_slots = compute_n_track_slots(tracks)

    per_jet = summarize_jets(tracks)  # event_id, jet_id, is_b_jet, n_tracks, n_muons

    if len(clusters):
        jetted_clusters = clusters[clusters["jet_id"] != -1]
        cluster_totals = jetted_clusters.groupby(["event_id", "jet_id"], as_index=False).agg(
            total_cluster_energy=("energy", "sum"), n_clusters=("particle_id", "count")
        )
    else:
        cluster_totals = pd.DataFrame(columns=["event_id", "jet_id", "total_cluster_energy", "n_clusters"])
    per_jet = per_jet.merge(cluster_totals, on=["event_id", "jet_id"], how="left")
    per_jet["total_cluster_energy"] = per_jet["total_cluster_energy"].fillna(0.0)
    per_jet["n_clusters"] = per_jet["n_clusters"].fillna(0).astype(int)

    track_groups = dict(tuple(jetted_tracks.groupby(["event_id", "jet_id"])))

    n_features = n_track_slots + len(SCALAR_FEATURE_NAMES)
    features = np.zeros((len(per_jet), n_features), dtype=np.float32)
    for row_index, row in enumerate(per_jet.itertuples(index=False)):
        d0_slots = _jet_d0_slots(track_groups[(row.event_id, row.jet_id)], n_track_slots)
        scalars = np.array(
            [row.n_tracks, row.n_muons, row.total_cluster_energy, row.n_clusters], dtype=np.float32
        )
        features[row_index] = np.concatenate([d0_slots, scalars])

    feature_names = [f"d0_track_{i}" for i in range(n_track_slots)] + list(SCALAR_FEATURE_NAMES)

    return Dataset(
        features=features,
        is_b_jet=per_jet["is_b_jet"].astype(int).to_numpy(),
        event_id=per_jet["event_id"].to_numpy(),
        jet_id=per_jet["jet_id"].to_numpy(),
        n_track_slots=n_track_slots,
        feature_names=feature_names,
    )


def compute_standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature `(mean, std)` from a training set -- the d0 slots,
    track/muon counts, and cluster energy sit on very different scales, and
    a small MLP trains far more reliably on standardized inputs. `std` is
    floored at a small epsilon so a constant feature (e.g. `n_muons` in a
    tiny/degenerate dataset) never divides by zero."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply a `(mean, std)` computed on some other (training) set --
    callers must reuse the training set's own values on an evaluation set,
    never recompute fresh ones on it, or the two would be scaled
    inconsistently."""
    return ((features - mean) / std).astype(np.float32)
