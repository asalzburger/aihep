import numpy as np
import pandas as pd
import pytest
from detectorreco2d.edm import CLUSTERS_COLUMNS, TRACKS_COLUMNS

from flavor_tagging.dataset import build_dataset, compute_n_track_slots, compute_standardization, standardize


def test_compute_n_track_slots_is_the_busiest_jets_track_count(synthetic_reco_run):
    tracks, _clusters = synthetic_reco_run
    n_slots = compute_n_track_slots(tracks)
    per_jet_counts = tracks[tracks["jet_id"] != -1].groupby(["event_id", "jet_id"]).size()
    assert n_slots == per_jet_counts.max()


def test_build_dataset_shape_and_labels(synthetic_reco_run):
    tracks, clusters = synthetic_reco_run
    dataset = build_dataset(tracks, clusters)

    n_jets = tracks[tracks["jet_id"] != -1].groupby(["event_id", "jet_id"]).ngroups
    assert dataset.features.shape == (n_jets, dataset.n_track_slots + 4)
    assert set(dataset.is_b_jet.tolist()) == {0, 1}
    assert len(dataset.feature_names) == dataset.features.shape[1]
    assert dataset.feature_names[-4:] == ["n_tracks", "n_muons", "total_cluster_energy", "n_clusters"]


def test_build_dataset_d0_slots_are_sorted_by_descending_pt_and_zero_padded():
    tracks = pd.DataFrame(
        [
            dict(
                event_id=0, particle_id=0, jet_id=0, is_b_jet=False, species="pi+", charge=1.0,
                d0_true=0, d0=1.0, phi0_true=0, phi0=0, pt_true=10, pt=10.0,
            ),
            dict(
                event_id=0, particle_id=1, jet_id=0, is_b_jet=False, species="pi+", charge=1.0,
                d0_true=0, d0=2.0, phi0_true=0, phi0=0, pt_true=50, pt=50.0,
            ),
        ],
        columns=TRACKS_COLUMNS,
    )
    clusters = pd.DataFrame(columns=CLUSTERS_COLUMNS)

    dataset = build_dataset(tracks, clusters, n_track_slots=4)
    # highest pt (50 -> d0=2.0) first, then pt=10 -> d0=1.0, then zero-padded
    np.testing.assert_allclose(dataset.features[0, :4], [2.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(dataset.features[0, 4:], [2, 0, 0.0, 0])  # n_tracks, n_muons, energy, n_clusters


def test_build_dataset_reuses_a_given_n_track_slots_and_truncates_excess():
    tracks = pd.DataFrame(
        [
            dict(
                event_id=0, particle_id=i, jet_id=0, is_b_jet=False, species="pi+", charge=1.0,
                d0_true=0, d0=float(i), phi0_true=0, phi0=0, pt_true=float(100 - i), pt=float(100 - i),
            )
            for i in range(5)
        ],
        columns=TRACKS_COLUMNS,
    )
    clusters = pd.DataFrame(columns=CLUSTERS_COLUMNS)

    dataset = build_dataset(tracks, clusters, n_track_slots=2)
    assert dataset.n_track_slots == 2
    # highest pt first: particle_id 0 (pt=100, d0=0.0), then 1 (pt=99, d0=1.0) -- the rest truncated
    np.testing.assert_allclose(dataset.features[0, :2], [0.0, 1.0])


def test_build_dataset_raises_on_jetless_tracks_without_an_explicit_n_track_slots():
    tracks = pd.DataFrame(columns=TRACKS_COLUMNS)
    clusters = pd.DataFrame(columns=CLUSTERS_COLUMNS)
    with pytest.raises(ValueError):
        build_dataset(tracks, clusters)


def test_compute_standardization_and_standardize():
    features = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]], dtype=np.float32)
    mean, std = compute_standardization(features)
    standardized = standardize(features, mean, std)

    assert standardized[:, 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert standardized[:, 0].std() == pytest.approx(1.0, abs=1e-6)
    # a constant feature has its std floored at 1.0 -- never divides by zero
    assert np.all(np.isfinite(standardized[:, 1]))
    assert standardized[0, 1] == pytest.approx(0.0)
