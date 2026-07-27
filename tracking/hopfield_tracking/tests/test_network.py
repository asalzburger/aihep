import numpy as np
import pandas as pd
import pytest

from hopfield_tracking.network import build_segments, mean_consecutive_hit_distance


def test_build_segments_respects_r_c_cutoff():
    x = np.array([0.0, 10.0, 100.0])
    y = np.array([0.0, 0.0, 0.0])
    segments = build_segments(x, y, r_c=15.0)
    pairs = {(s.start_id, s.end_id) for s in segments}
    assert pairs == {(0, 1), (1, 0)}  # point 2 is too far from both 0 and 1


def test_build_segments_excludes_same_layer_pairs():
    # two hits on the same layer (id 0), close together, plus one hit on a
    # different layer (id 1) -- only cross-layer segments are physical.
    x = np.array([0.0, 5.0, 0.0])
    y = np.array([0.0, 0.0, 50.0])
    layer_ids = np.array([0, 0, 1])
    segments = build_segments(x, y, r_c=60.0, layer_ids=layer_ids)
    pairs = {(s.start_id, s.end_id) for s in segments}
    assert (0, 1) not in pairs and (1, 0) not in pairs  # same layer, excluded
    assert (0, 2) in pairs and (2, 0) in pairs  # different layers, kept


def test_build_segments_keeps_same_layer_pairs_without_layer_ids():
    # default behavior (no layer_ids given) is unchanged: pure distance cutoff
    x = np.array([0.0, 5.0])
    y = np.array([0.0, 0.0])
    segments = build_segments(x, y, r_c=60.0)
    assert len(segments) == 2


def test_build_segments_generates_both_directions():
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    segments = build_segments(x, y, r_c=15.0)
    assert len(segments) == 2
    pairs = {(s.start_id, s.end_id) for s in segments}
    assert pairs == {(0, 1), (1, 0)}


def test_build_segments_indices_are_sequential():
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 0.0, 0.0])
    segments = build_segments(x, y, r_c=100.0)
    assert [s.index for s in segments] == list(range(len(segments)))


def test_segment_vector_and_length():
    x = np.array([0.0, 3.0])
    y = np.array([0.0, 4.0])
    segments = build_segments(x, y, r_c=10.0)
    seg = next(s for s in segments if s.start_id == 0)
    assert seg.vector == pytest.approx((3.0, 4.0))
    assert seg.length == pytest.approx(5.0)


def test_mean_consecutive_hit_distance_on_evenly_spaced_track():
    hits = pd.DataFrame({"particle_id": [0, 0, 0, 0], "x": [0.0, 10.0, 20.0, 30.0], "y": [0.0, 0.0, 0.0, 0.0]})
    assert mean_consecutive_hit_distance(hits) == pytest.approx(10.0)


def test_mean_consecutive_hit_distance_averages_across_tracks():
    hits = pd.DataFrame(
        {
            "particle_id": [0, 0, 1, 1],
            "x": [0.0, 10.0, 0.0, 20.0],
            "y": [0.0, 0.0, 0.0, 0.0],
        }
    )
    assert mean_consecutive_hit_distance(hits) == pytest.approx((10.0 + 20.0) / 2)


def test_mean_consecutive_hit_distance_raises_without_multi_point_groups():
    hits = pd.DataFrame({"particle_id": [0, 1], "x": [0.0, 1.0], "y": [0.0, 1.0]})
    with pytest.raises(ValueError):
        mean_consecutive_hit_distance(hits)
