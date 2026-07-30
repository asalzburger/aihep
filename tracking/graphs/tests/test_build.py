import math

import pandas as pd
import pytest
from tracksim2d.edm import HITS_COLUMNS

from graphs.build import _wrap_phi, build_edges, build_graph
from graphs.edm import TrackGraph
from graphs.prescription import ConnectionRules, FullyConnected, Regional


def _hit(event_id, hit_id, x, y, layer_id, particle_id=0):
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


def test_fully_connected_undirected_gives_one_edge_per_unordered_pair():
    hits = _hits([_hit(0, i, float(i), 0.0, layer_id=i) for i in range(4)])
    edges = build_edges(hits, FullyConnected())
    assert len(edges) == 4 * 3 // 2  # C(4, 2)
    pairs = {frozenset((row.src_hit_id, row.dst_hit_id)) for row in edges.itertuples()}
    assert len(pairs) == len(edges)  # no duplicate unordered pairs


def test_fully_connected_directed_gives_both_orderings():
    hits = _hits([_hit(0, i, float(i), 0.0, layer_id=i) for i in range(4)])
    edges = build_edges(hits, FullyConnected(directed=True))
    assert len(edges) == 4 * 3  # every ordered pair
    directed_pairs = {(row.src_hit_id, row.dst_hit_id) for row in edges.itertuples()}
    assert (0, 1) in directed_pairs and (1, 0) in directed_pairs


def test_regional_only_connects_hits_in_the_same_phi_sector():
    # phi_width = pi/2 -> 4 quadrant sectors. Two hits near phi=0 (same
    # sector), one hit near phi=pi (a different sector).
    hits = _hits(
        [
            _hit(0, 0, 10.0, 0.1, layer_id=0),  # phi ~ 0
            _hit(0, 1, 10.0, 0.2, layer_id=1),  # phi ~ 0, same sector as hit 0
            _hit(0, 2, -10.0, 0.1, layer_id=2),  # phi ~ pi, different sector
        ]
    )
    edges = build_edges(hits, Regional(phi_width=math.pi / 2))
    pairs = {frozenset((row.src_hit_id, row.dst_hit_id)) for row in edges.itertuples()}
    assert frozenset((0, 1)) in pairs
    assert frozenset((0, 2)) not in pairs
    assert frozenset((1, 2)) not in pairs


def test_connection_rules_delta_layer_id_restricts_to_adjacent_layers():
    hits = _hits([_hit(0, i, x=float(i) * 10.0, y=0.0, layer_id=i) for i in range(4)])  # layers 0,1,2,3 outward
    edges = build_edges(hits, ConnectionRules(delta_layer_id=(1.0, 1.0)))
    pairs = {(row.src_hit_id, row.dst_hit_id) for row in edges.itertuples()}
    # only adjacent, outward (src layer -> src layer + 1) pairs survive
    assert pairs == {(0, 1), (1, 2), (2, 3)}


def test_connection_rules_delta_r_restricts_by_radius_change():
    hits = _hits(
        [
            _hit(0, 0, x=10.0, y=0.0, layer_id=0),  # r=10
            _hit(0, 1, x=15.0, y=0.0, layer_id=1),  # r=15, delta_r=5 from hit 0
            _hit(0, 2, x=200.0, y=0.0, layer_id=2),  # r=200, delta_r=190 from hit 0
        ]
    )
    edges = build_edges(hits, ConnectionRules(delta_r=(0.0, 10.0)))
    pairs = {(row.src_hit_id, row.dst_hit_id) for row in edges.itertuples()}
    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (2, 0) not in pairs


def test_connection_rules_delta_phi_wraps_across_the_pi_seam():
    # two hits straddling the +-pi seam: phi ~ +3.13 and phi ~ -3.13, a true
    # angular separation of ~0.02 rad, not the ~6.26 rad a naive subtraction
    # would give.
    phi_a, phi_b = 3.13, -3.13
    hits = _hits(
        [
            _hit(0, 0, x=10.0 * math.cos(phi_a), y=10.0 * math.sin(phi_a), layer_id=0),
            _hit(0, 1, x=10.0 * math.cos(phi_b), y=10.0 * math.sin(phi_b), layer_id=1),
        ]
    )
    edges = build_edges(hits, ConnectionRules(delta_phi=(-0.1, 0.1)))
    # both directions satisfy the symmetric range; without wrapping, the raw
    # subtraction (~+-6.26 rad) would fall outside it and give 0 edges
    assert len(edges) == 2
    pairs = {(row.src_hit_id, row.dst_hit_id) for row in edges.itertuples()}
    assert pairs == {(0, 1), (1, 0)}


def test_hits_in_different_events_never_connect():
    # 2 hits in each of 2 events, at identical (x, y) across events -- if
    # event boundaries weren't respected, FullyConnected would produce
    # C(4, 2)=6 edges instead of 1 per event (2 total).
    hits = _hits(
        [
            _hit(0, 0, 0.0, 0.0, layer_id=0),
            _hit(0, 1, 1.0, 0.0, layer_id=1),
            _hit(1, 2, 0.0, 0.0, layer_id=0),
            _hit(1, 3, 1.0, 0.0, layer_id=1),
        ]
    )
    edges = build_edges(hits, FullyConnected())
    assert len(edges) == 2
    for row in edges.itertuples():
        assert {row.src_hit_id, row.dst_hit_id} <= ({0, 1} if row.event_id == 0 else {2, 3})


def test_edge_feature_values_are_dst_minus_src():
    hits = _hits([_hit(0, 0, x=1.0, y=0.0, layer_id=0), _hit(0, 1, x=0.0, y=1.0, layer_id=2)])
    edges = build_edges(hits, FullyConnected())
    assert len(edges) == 1
    edge = edges.iloc[0]
    assert edge["src_hit_id"] == 0 and edge["dst_hit_id"] == 1
    assert edge["delta_layer_id"] == 2
    assert edge["delta_r"] == pytest.approx(0.0)  # both at r=1
    assert edge["delta_phi"] == pytest.approx(math.pi / 2)
    assert edge["delta_x"] == pytest.approx(-1.0)
    assert edge["delta_y"] == pytest.approx(1.0)
    assert edge["distance"] == pytest.approx(math.sqrt(2.0))


def test_wrap_phi_handles_the_seam_and_stays_in_range():
    assert _wrap_phi(0.0) == pytest.approx(0.0)
    assert _wrap_phi(2.0 * math.pi - 0.01) == pytest.approx(-0.01)
    assert _wrap_phi(-2.0 * math.pi + 0.01) == pytest.approx(0.01)


def test_build_graph_bundles_nodes_and_edges():
    hits = _hits([_hit(0, i, float(i), 0.0, layer_id=i) for i in range(3)])
    graph = build_graph(hits, FullyConnected())
    assert isinstance(graph, TrackGraph)
    assert graph.nodes is hits
    assert len(graph.edges) == 3
