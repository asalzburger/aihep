import pandas as pd
import pytest

from tracker.clustering import cluster_hits
from tracker.edm import HITS_COLUMNS
from tracker.io import read_run, write_run


def _hit(cell_index, hit_id):
    return dict(
        event_id=0,
        particle_id=0,
        layer_id=0,
        hit_id=hit_id,
        x=float(hit_id),
        y=0.0,
        s_local=float(cell_index),
        path_length=1.0,
        cell_index=cell_index,
    )


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_write_read_round_trip(tmp_path, fmt):
    hits = pd.DataFrame([_hit(0, 0), _hit(1, 1), _hit(5, 2)], columns=HITS_COLUMNS)
    hits_out, clusters = cluster_hits(hits)

    write_run(tmp_path, fmt, hits_out, clusters)
    hits2, clusters2 = read_run(tmp_path, fmt)

    pd.testing.assert_frame_equal(hits_out, hits2, check_dtype=False)
    pd.testing.assert_frame_equal(clusters, clusters2, check_dtype=False)
