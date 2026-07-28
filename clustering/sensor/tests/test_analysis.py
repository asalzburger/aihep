import pandas as pd
import pytest

from sensor.analysis import MATCHED_COLUMNS, compute_residuals, match_clusters_to_truth
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
