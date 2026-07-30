import math

import pandas as pd
import pytest
from tracksim2d.edm import HITS_COLUMNS

from graphs.build import build_graph
from graphs.io import read_edges, read_graph, write_edges, write_graph
from graphs.prescription import FullyConnected


def _hit(event_id, hit_id, x, y, layer_id):
    return dict(
        event_id=event_id, particle_id=0, layer_id=layer_id, hit_id=hit_id, x=x, y=y, s_local=0.0,
        path_length=math.hypot(x, y),
    )


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_edges_round_trip(tmp_path, fmt):
    hits = pd.DataFrame([_hit(0, i, float(i), 0.0, i) for i in range(3)], columns=HITS_COLUMNS)
    graph = build_graph(hits, FullyConnected())

    path = tmp_path / f"edges.{fmt}"
    write_edges(path, graph.edges, fmt)
    round_tripped = read_edges(path, fmt)

    pd.testing.assert_frame_equal(round_tripped, graph.edges, check_dtype=False)


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_write_graph_and_read_graph_round_trip(tmp_path, fmt):
    hits = pd.DataFrame([_hit(0, i, float(i), 0.0, i) for i in range(3)], columns=HITS_COLUMNS)
    graph = build_graph(hits, FullyConnected())

    paths = write_graph(tmp_path, fmt, graph)
    assert paths["edges"].exists()

    read_back = read_graph(tmp_path, fmt, nodes=hits)
    assert read_back.nodes is hits
    pd.testing.assert_frame_equal(read_back.edges, graph.edges, check_dtype=False)
