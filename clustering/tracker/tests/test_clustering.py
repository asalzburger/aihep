import math

import pandas as pd
import pytest

from tracker.clustering import cluster_hits
from tracker.edm import HITS_COLUMNS


def _hit(cell_index, layer_id=0, event_id=0, hit_id=0, s_local=0.0, x=0.0, y=0.0):
    return dict(
        event_id=event_id,
        particle_id=0,
        layer_id=layer_id,
        hit_id=hit_id,
        x=x,
        y=y,
        s_local=s_local,
        path_length=1.0,
        cell_index=cell_index,
    )


def test_adjacent_cells_form_a_single_cluster():
    hits = pd.DataFrame([_hit(0, hit_id=0), _hit(1, hit_id=1), _hit(2, hit_id=2)], columns=HITS_COLUMNS)
    hits_out, clusters = cluster_hits(hits)
    assert len(clusters) == 1
    assert clusters.iloc[0]["n_cells"] == 3
    assert clusters.iloc[0]["n_hits"] == 3
    assert set(hits_out["cluster_id"]) == {0}


def test_a_gap_splits_into_two_clusters():
    hits = pd.DataFrame(
        [_hit(0, hit_id=0), _hit(1, hit_id=1), _hit(5, hit_id=2), _hit(6, hit_id=3)], columns=HITS_COLUMNS
    )
    hits_out, clusters = cluster_hits(hits, connectivity_gap=1)
    assert len(clusters) == 2
    assert sorted(clusters["n_cells"]) == [2, 2]
    assert set(hits_out["cluster_id"]) == {0, 1}


def test_cluster_ids_unique_across_layers_within_an_event():
    hits = pd.DataFrame(
        [_hit(0, layer_id=0, hit_id=0), _hit(1, layer_id=0, hit_id=1), _hit(0, layer_id=1, hit_id=2)],
        columns=HITS_COLUMNS,
    )
    hits_out, clusters = cluster_hits(hits)
    assert len(clusters) == 2
    assert set(clusters["cluster_id"]) == {0, 1}
    assert set(clusters["layer_id"]) == {0, 1}


def test_hits_with_no_cell_index_are_unclustered():
    hits = pd.DataFrame([_hit(math.nan, hit_id=0), _hit(0, hit_id=1)], columns=HITS_COLUMNS)
    hits_out, clusters = cluster_hits(hits)
    unclustered = hits_out[hits_out["hit_id"] == 0]
    assert unclustered.iloc[0]["cluster_id"] == -1
    assert len(clusters) == 1  # only the digitized hit forms a cluster


def test_empty_hits_returns_empty_clusters():
    hits = pd.DataFrame(columns=HITS_COLUMNS)
    hits_out, clusters = cluster_hits(hits)
    assert hits_out.empty
    assert clusters.empty


def test_centroid_is_the_mean_position_of_the_clusters_hits():
    hits = pd.DataFrame(
        [_hit(0, hit_id=0, s_local=10.0, x=1.0, y=2.0), _hit(1, hit_id=1, s_local=20.0, x=3.0, y=4.0)],
        columns=HITS_COLUMNS,
    )
    _, clusters = cluster_hits(hits)
    row = clusters.iloc[0]
    assert row["s_centroid"] == pytest.approx(15.0)
    assert row["x_centroid"] == pytest.approx(2.0)
    assert row["y_centroid"] == pytest.approx(3.0)
