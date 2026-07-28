import pandas as pd
import pytest

from sensor.sim.clustering import cluster_hits
from sensor.sim.config import DetectorConfig


def _detector():
    return DetectorConfig(thickness_um=150, pitch_x_um=25, pitch_y_um=50, n_pixels_x=10, n_pixels_y=10)


def _hits(rows):
    return pd.DataFrame(rows, columns=["event_id", "ix", "iy", "x_center_um", "y_center_um", "charge"])


def test_readout_threshold_drops_low_charge_pixels_before_clustering():
    detector = _detector()
    hits = _hits(
        [
            dict(event_id=0, ix=0, iy=0, x_center_um=12.5, y_center_um=25, charge=1.0),
            dict(event_id=0, ix=1, iy=0, x_center_um=37.5, y_center_um=25, charge=0.1),
        ]
    )

    hits_out, clusters = cluster_hits(hits, detector, readout_threshold=0.0)
    assert len(hits_out) == 2
    assert clusters.iloc[0]["n_pixels"] == 2  # both pixels survive, joined into one cluster

    hits_out, clusters = cluster_hits(hits, detector, readout_threshold=0.15)
    assert len(hits_out) == 1
    assert len(clusters) == 1
    assert clusters.iloc[0]["n_pixels"] == 1
    assert clusters.iloc[0]["charge_sum"] == pytest.approx(1.0)


def test_readout_threshold_cutting_everything_returns_empty():
    detector = _detector()
    hits = _hits([dict(event_id=0, ix=0, iy=0, x_center_um=12.5, y_center_um=25, charge=0.05)])

    hits_out, clusters = cluster_hits(hits, detector, readout_threshold=0.15)
    assert hits_out.empty
    assert clusters.empty


def test_digital_and_charge_centroids_differ_for_asymmetric_charge():
    detector = _detector()
    # Two adjacent pixels in x with asymmetric charge: the charge-weighted
    # centroid should be pulled toward the higher-charge pixel, while the
    # digital centroid sits exactly between the two pixel centers.
    hits = _hits(
        [
            dict(event_id=0, ix=0, iy=0, x_center_um=12.5, y_center_um=25, charge=1.0),
            dict(event_id=0, ix=1, iy=0, x_center_um=37.5, y_center_um=25, charge=3.0),
        ]
    )
    _, clusters = cluster_hits(hits, detector)
    row = clusters.iloc[0]

    assert row["x_centroid_digital_um"] == pytest.approx((12.5 + 37.5) / 2)
    assert row["x_centroid_um"] == pytest.approx((12.5 * 1.0 + 37.5 * 3.0) / 4.0)
    assert row["x_centroid_um"] > row["x_centroid_digital_um"]


def test_digital_and_charge_centroids_match_for_single_pixel_cluster():
    detector = _detector()
    hits = _hits([dict(event_id=0, ix=0, iy=0, x_center_um=12.5, y_center_um=25, charge=2.0)])
    _, clusters = cluster_hits(hits, detector)
    row = clusters.iloc[0]

    assert row["x_centroid_digital_um"] == pytest.approx(row["x_centroid_um"])
    assert row["y_centroid_digital_um"] == pytest.approx(row["y_centroid_um"])
