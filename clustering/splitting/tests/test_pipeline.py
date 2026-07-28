import pandas as pd
import pytest

from splitting.base import Splitter
from splitting.pipeline import apply_splitter


class _KeySplitter(Splitter):
    """Test double: the split key is read straight from a hits column,
    so tests can drive `apply_splitter`'s generic renumbering/aggregation
    machinery without any particular splitting algorithm."""

    name = "test-key"

    def split_key(self, hits: pd.DataFrame, clusters: pd.DataFrame, contributions: pd.DataFrame) -> pd.Series:
        return hits["key"]


def _hit(event_id, ix, iy, cluster_id, charge, key):
    return dict(
        event_id=event_id,
        ix=ix,
        iy=iy,
        x_center_um=(ix + 0.5) * 25.0,
        y_center_um=(iy + 0.5) * 50.0,
        charge=charge,
        cluster_id=cluster_id,
        key=key,
    )


def test_apply_splitter_splits_a_merged_cluster_by_key():
    hits = pd.DataFrame(
        [
            _hit(0, 0, 0, cluster_id=0, charge=1.0, key="a"),
            _hit(0, 1, 0, cluster_id=0, charge=1.0, key="a"),
            _hit(0, 2, 0, cluster_id=0, charge=2.0, key="b"),
        ]
    )
    new_hits, new_clusters = apply_splitter(_KeySplitter(), hits, pd.DataFrame(), pd.DataFrame())

    assert len(new_hits) == 3  # no pixels dropped
    assert sorted(new_hits["cluster_id"].unique()) == [0, 1]
    assert len(new_clusters) == 2

    by_cluster = new_clusters.set_index("cluster_id")
    assert by_cluster["n_pixels"].sum() == 3
    assert by_cluster["charge_sum"].sum() == pytest.approx(4.0)
    # key "a" cluster: two equal-charge pixels -> charge-weighted == digital centroid
    a_cluster = by_cluster[by_cluster["n_pixels"] == 2].iloc[0]
    assert a_cluster["x_centroid_um"] == pytest.approx(a_cluster["x_centroid_digital_um"])


def test_apply_splitter_is_a_no_op_for_a_constant_key():
    hits = pd.DataFrame(
        [
            _hit(0, 0, 0, cluster_id=0, charge=1.0, key="same"),
            _hit(0, 5, 5, cluster_id=1, charge=2.0, key="same"),
            _hit(1, 0, 0, cluster_id=0, charge=1.5, key="same"),
        ]
    )
    new_hits, new_clusters = apply_splitter(_KeySplitter(), hits, pd.DataFrame(), pd.DataFrame())

    # a constant key never splits anything apart: cluster_id comes back
    # exactly as it went in (already dense/0-based per event)
    left = hits.sort_values(["event_id", "ix", "iy"])["cluster_id"].reset_index(drop=True)
    right = new_hits.sort_values(["event_id", "ix", "iy"])["cluster_id"].reset_index(drop=True)
    pd.testing.assert_series_equal(left, right)
    assert len(new_clusters) == 3


def test_apply_splitter_rejects_mismatched_label_length():
    class _BadSplitter(Splitter):
        name = "bad"

        def split_key(self, hits, clusters, contributions):
            return pd.Series([0])  # wrong length

    hits = pd.DataFrame(
        [
            _hit(0, 0, 0, cluster_id=0, charge=1.0, key="a"),
            _hit(0, 1, 0, cluster_id=0, charge=1.0, key="a"),
        ]
    )
    with pytest.raises(ValueError):
        apply_splitter(_BadSplitter(), hits, pd.DataFrame(), pd.DataFrame())
