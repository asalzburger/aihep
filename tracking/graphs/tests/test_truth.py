import math

import pandas as pd
import pytest
from detectorsim2d.edm import HITS_COLUMNS

from graphs.build import build_graph
from graphs.edm import LABELED_EDGES_COLUMNS
from graphs.prescription import FullyConnected
from graphs.truth import label_edges, label_graph, purity


def _hit(event_id, hit_id, x, y, layer_id, particle_id):
    return dict(
        event_id=event_id,
        particle_id=particle_id,
        layer_id=layer_id,
        hit_id=hit_id,
        x=x,
        y=y,
        s_local=0.0,
        path_length=math.hypot(x, y),
    )


def _hits(rows):
    return pd.DataFrame(rows, columns=HITS_COLUMNS)


def test_label_edges_marks_same_particle_pairs_true():
    # particle 0: hits 0,1 (a true segment); particle 1: hit 2 (fans off
    # nearby, close in (x, y) to hit 1, but a different particle)
    hits = _hits(
        [
            _hit(0, 0, 10.0, 0.0, layer_id=0, particle_id=0),
            _hit(0, 1, 20.0, 0.0, layer_id=1, particle_id=0),
            _hit(0, 2, 20.0, 0.1, layer_id=1, particle_id=1),
        ]
    )
    edges = build_graph(hits, FullyConnected()).edges
    labeled = label_edges(hits, edges)

    assert list(labeled.columns) == LABELED_EDGES_COLUMNS
    by_pair = {frozenset((row.src_hit_id, row.dst_hit_id)): bool(row.is_true_edge) for row in labeled.itertuples()}
    assert by_pair[frozenset((0, 1))] is True
    assert by_pair[frozenset((0, 2))] is False
    assert by_pair[frozenset((1, 2))] is False


def test_label_edges_does_not_mutate_the_input():
    hits = _hits([_hit(0, 0, 1.0, 0.0, layer_id=0, particle_id=0), _hit(0, 1, 2.0, 0.0, layer_id=1, particle_id=0)])
    edges = build_graph(hits, FullyConnected()).edges
    original_columns = list(edges.columns)

    label_edges(hits, edges)

    assert list(edges.columns) == original_columns  # unlabeled edges untouched


def test_label_graph_wraps_label_edges_and_keeps_nodes():
    hits = _hits([_hit(0, 0, 1.0, 0.0, layer_id=0, particle_id=0), _hit(0, 1, 2.0, 0.0, layer_id=1, particle_id=1)])
    graph = build_graph(hits, FullyConnected())
    labeled_graph = label_graph(graph)

    assert labeled_graph.nodes is graph.nodes
    assert "is_true_edge" in labeled_graph.edges.columns
    assert not labeled_graph.edges.iloc[0]["is_true_edge"]  # different particles


def test_purity_computes_fraction_of_true_edges():
    hits = _hits(
        [
            _hit(0, 0, 1.0, 0.0, layer_id=0, particle_id=0),
            _hit(0, 1, 2.0, 0.0, layer_id=1, particle_id=0),  # true with hit 0
            _hit(0, 2, 3.0, 0.0, layer_id=2, particle_id=1),  # false with both
        ]
    )
    edges = build_graph(hits, FullyConnected()).edges  # C(3,2)=3 edges: 1 true, 2 false
    labeled = label_edges(hits, edges)
    assert purity(labeled) == pytest.approx(1.0 / 3.0)


def test_purity_raises_without_truth_column():
    hits = _hits([_hit(0, 0, 1.0, 0.0, layer_id=0, particle_id=0), _hit(0, 1, 2.0, 0.0, layer_id=1, particle_id=0)])
    edges = build_graph(hits, FullyConnected()).edges
    with pytest.raises(ValueError):
        purity(edges)


def test_purity_of_empty_edges_is_nan():
    hits = _hits([_hit(0, 0, 1.0, 0.0, layer_id=0, particle_id=0)])
    edges = build_graph(hits, FullyConnected()).edges  # 1 hit -> 0 edges
    labeled = label_edges(hits, edges)
    assert math.isnan(purity(labeled))
