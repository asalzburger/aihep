import pandas as pd
import pytest

from sensor.analysis import (
    MATCHED_COLUMNS,
    cluster_purity,
    compute_residuals,
    dominant_cluster_per_particle,
    dominant_particle_per_cluster,
    match_clusters_to_truth,
)
from sensor.sim.config import DetectorConfig

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


def _detector():
    return DetectorConfig(thickness_um=150.0)


def _truth_row(**overrides):
    row = dict(
        event_id=0, particle_id=0, x0_um=0.0, y0_um=0.0, dxdz=0.0, dydz=0.0,
        charge_deposited=1.0, path_length_um=150.0,
    )
    row.update(overrides)
    return row


def _cluster_row(**overrides):
    row = dict(
        event_id=0, cluster_id=0, n_pixels=1, charge_sum=1.0,
        x_centroid_um=0.0, y_centroid_um=0.0, x_centroid_digital_um=0.0, y_centroid_digital_um=0.0,
        x_span_pixels=1, y_span_pixels=1,
    )
    row.update(overrides)
    return row


def test_perpendicular_track_zero_residual_both_types():
    detector = _detector()
    truth = pd.DataFrame([_truth_row(x0_um=100.0, y0_um=200.0)])
    clusters = pd.DataFrame(
        [_cluster_row(x_centroid_um=100.0, y_centroid_um=200.0, x_centroid_digital_um=100.0, y_centroid_digital_um=200.0)]
    )

    for centroid_type in ("charge", "digital"):
        residuals = compute_residuals(clusters, truth, detector, type=centroid_type)
        assert len(residuals) == 1
        assert residuals.iloc[0]["residual_x_um"] == pytest.approx(0.0)
        assert residuals.iloc[0]["residual_y_um"] == pytest.approx(0.0)


def test_matches_nearest_cluster_in_multi_cluster_event():
    detector = _detector()
    truth = pd.DataFrame([_truth_row(x0_um=0.0, y0_um=0.0)])
    clusters = pd.DataFrame(
        [
            _cluster_row(cluster_id=0, x_centroid_um=5.0, y_centroid_um=5.0, x_centroid_digital_um=5.0, y_centroid_digital_um=5.0),
            _cluster_row(cluster_id=1, x_centroid_um=500.0, y_centroid_um=500.0, x_centroid_digital_um=500.0, y_centroid_digital_um=500.0),
        ]
    )

    matched = match_clusters_to_truth(clusters, truth, detector, type="charge")
    assert list(matched.columns) == MATCHED_COLUMNS
    assert len(matched) == 1
    assert matched.iloc[0]["cluster_id"] == 0


def test_truth_without_clusters_in_event_is_dropped():
    detector = _detector()
    truth = pd.DataFrame([_truth_row()])
    clusters = pd.DataFrame(columns=CLUSTERS_COLUMNS)

    matched = match_clusters_to_truth(clusters, truth, detector, type="charge")
    assert matched.empty


def test_residual_reflects_offset_between_recon_and_true():
    detector = _detector()
    truth = pd.DataFrame([_truth_row(x0_um=0.0, y0_um=0.0)])
    clusters = pd.DataFrame(
        [_cluster_row(x_centroid_um=3.0, y_centroid_um=-2.0, x_centroid_digital_um=10.0, y_centroid_digital_um=10.0)]
    )

    charge_residuals = compute_residuals(clusters, truth, detector, type="charge")
    assert charge_residuals.iloc[0]["residual_x_um"] == pytest.approx(3.0)
    assert charge_residuals.iloc[0]["residual_y_um"] == pytest.approx(-2.0)

    digital_residuals = compute_residuals(clusters, truth, detector, type="digital")
    assert digital_residuals.iloc[0]["residual_x_um"] == pytest.approx(10.0)
    assert digital_residuals.iloc[0]["residual_y_um"] == pytest.approx(10.0)


def _swapped_contribution_scenario():
    """Two clusters (pixels 0/1 -> cluster 0, pixels 10/11 -> cluster 1) and
    two truth particles whose deposited charge (contributions) lands in the
    *opposite* cluster from the one nearest their true position — the case
    where nearest-position matching gets it wrong and the exact
    contribution-based match is needed."""
    detector = _detector()
    hits = pd.DataFrame(
        [
            dict(event_id=0, ix=0, iy=0, cluster_id=0),
            dict(event_id=0, ix=1, iy=0, cluster_id=0),
            dict(event_id=0, ix=10, iy=0, cluster_id=1),
            dict(event_id=0, ix=11, iy=0, cluster_id=1),
        ]
    )
    contributions = pd.DataFrame(
        [
            # particle 0's charge actually lands in cluster 1's pixels...
            dict(event_id=0, particle_id=0, ix=10, iy=0, charge=1.0),
            dict(event_id=0, particle_id=0, ix=11, iy=0, charge=1.0),
            # ...and particle 1's charge actually lands in cluster 0's pixels.
            dict(event_id=0, particle_id=1, ix=0, iy=0, charge=1.0),
            dict(event_id=0, particle_id=1, ix=1, iy=0, charge=1.0),
        ]
    )
    clusters = pd.DataFrame(
        [
            _cluster_row(cluster_id=0, x_centroid_um=25.0, y_centroid_um=25.0, x_centroid_digital_um=25.0, y_centroid_digital_um=25.0),
            _cluster_row(cluster_id=1, x_centroid_um=275.0, y_centroid_um=25.0, x_centroid_digital_um=275.0, y_centroid_digital_um=25.0),
        ]
    )
    # true positions close to the "wrong" cluster (0 near cluster 0, 1 near cluster 1)
    truth = pd.DataFrame(
        [
            _truth_row(particle_id=0, x0_um=100.0, y0_um=25.0),
            _truth_row(particle_id=1, x0_um=200.0, y0_um=25.0),
        ]
    )
    return detector, hits, contributions, clusters, truth


def test_cluster_purity_reports_dominant_and_fraction():
    _, hits, contributions, _, _ = _swapped_contribution_scenario()

    purity = cluster_purity(hits, contributions)
    assert len(purity) == 2  # one particle per cluster here, no overlap
    assert set(purity["fraction"]) == {1.0}

    dominant = dominant_particle_per_cluster(hits, contributions)
    assert dict(zip(dominant["cluster_id"], dominant["particle_id"])) == {0: 1, 1: 0}

    dominant_cluster = dominant_cluster_per_particle(hits, contributions)
    assert dict(zip(dominant_cluster["particle_id"], dominant_cluster["cluster_id"])) == {0: 1, 1: 0}


def test_nearest_position_matching_picks_wrong_cluster():
    detector, _, _, clusters, truth = _swapped_contribution_scenario()

    matched = match_clusters_to_truth(clusters, truth, detector, type="charge")
    matched = matched.set_index("particle_id")
    # nearest-position matching is fooled: particle 0 (true x=100) is closer
    # to cluster 0 (x=25) than cluster 1 (x=275), and vice versa for particle 1
    assert matched.loc[0, "cluster_id"] == 0
    assert matched.loc[1, "cluster_id"] == 1


def test_contribution_based_matching_is_exact():
    detector, hits, contributions, clusters, truth = _swapped_contribution_scenario()

    matched = match_clusters_to_truth(
        clusters, truth, detector, type="charge", hits=hits, contributions=contributions
    )
    matched = matched.set_index("particle_id")
    # with the exact charge link, particle 0 correctly matches cluster 1
    # (where its charge actually went) and particle 1 matches cluster 0
    assert matched.loc[0, "cluster_id"] == 1
    assert matched.loc[1, "cluster_id"] == 0
