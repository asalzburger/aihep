import pandas as pd

from hopfield_tracking.cli import run


def test_run_auto_excludes_same_layer_segments_when_layer_id_present():
    # particle 0: two hits, layers 0 and 1, 50 apart -- a real track segment.
    # particle 1: one hit on layer 0, right next to particle 0's layer-0 hit
    # -- close enough in (x, y) that without layer awareness it would be a
    # spurious same-layer candidate segment.
    hits = pd.DataFrame(
        {
            "particle_id": [0, 0, 1],
            "layer_id": [0, 1, 0],
            "x": [0.0, 0.0, 5.0],
            "y": [0.0, 50.0, 0.0],
        }
    )
    segments, _, _, _ = run(hits, r_c=60.0, r_scale=50.0, seed=0)
    same_layer = [s for s in segments if hits.loc[s.start_id, "layer_id"] == hits.loc[s.end_id, "layer_id"]]
    assert same_layer == []


def test_run_without_layer_id_column_still_works():
    hits = pd.DataFrame({"particle_id": [0, 0], "x": [0.0, 0.0], "y": [0.0, 50.0]})
    segments, history, chains, score = run(hits, r_c=60.0, r_scale=50.0, seed=0)
    assert len(segments) == 2  # both directions, no layer filtering applied
    assert score["n_true_tracks"] == 1
