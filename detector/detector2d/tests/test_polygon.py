import math

import pytest
from detector2d.geometry import Trajectory
from detector2d.intersect import first_intersection
from detector2d.polygon import (
    build_muon_system,
    build_polygon,
    build_polygon_triplet_station,
    polygon_vertices,
)


def _side_midpoint_radius(side):
    (x1, y1), (x2, y2) = side.p1, side.p2
    return math.hypot(0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _is_parallel(a, b):
    """Undirected parallelism, via the cross product of the two directions --
    an angle comparison would trip over the pi/-pi branch cut."""
    (ax, ay), (bx, by) = a.direction, b.direction
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    return abs(ax / na * by / nb - ay / na * bx / nb) < 1e-12


def test_octagon_has_eight_sides_sharing_one_layer_id():
    sides = build_polygon(layer_id=300, apothem=500.0, n_sides=8)
    assert len(sides) == 8
    assert {s.layer_id for s in sides} == {300}
    assert {s.system for s in sides} == {"muon"}


def test_every_side_midpoint_sits_at_the_apothem():
    """The apothem is defined as origin-to-side-midpoint distance -- the
    closest approach of the polygon, not its circumradius."""
    for side in build_polygon(layer_id=0, apothem=520.0, n_sides=8):
        assert _side_midpoint_radius(side) == pytest.approx(520.0)


def test_vertices_are_equidistant_and_on_the_circumradius():
    apothem, n_sides = 520.0, 8
    vertices = polygon_vertices(apothem, n_sides)
    circumradius = apothem / math.cos(math.pi / n_sides)
    assert len(vertices) == n_sides
    assert all(math.hypot(x, y) == pytest.approx(circumradius) for x, y in vertices)

    edges = [
        math.dist(vertices[i], vertices[(i + 1) % n_sides]) for i in range(n_sides)
    ]
    assert edges == pytest.approx([edges[0]] * n_sides)


def test_a_triplet_is_three_parallel_planes_one_gap_apart():
    layers = build_polygon_triplet_station(layer_id_base=300, apothem=520.0, gap=8.0, n_planes=3)
    assert len(layers) == 3 * 8
    assert sorted({layer.layer_id for layer in layers}) == [300, 301, 302]

    by_plane = {plane: [l for l in layers if l.layer_id == plane] for plane in (300, 301, 302)}
    for plane, expected_apothem in zip((300, 301, 302), (520.0, 528.0, 536.0)):
        assert all(
            _side_midpoint_radius(side) == pytest.approx(expected_apothem) for side in by_plane[plane]
        )
    # side i of each plane is parallel to side i of every other plane
    for i in range(8):
        assert _is_parallel(by_plane[300][i], by_plane[301][i])
        assert _is_parallel(by_plane[300][i], by_plane[302][i])


def test_three_stations_are_equally_spaced_with_non_colliding_layer_ids():
    layers = build_muon_system(
        layer_id_base=300, apothem_inner=520.0, station_spacing=100.0, n_stations=3, n_planes=3
    )
    assert len(layers) == 3 * 3 * 8  # stations x planes x sides
    assert sorted({layer.layer_id for layer in layers}) == [300, 301, 302, 310, 311, 312, 320, 321, 322]

    innermost_apothems = [
        _side_midpoint_radius(next(l for l in layers if l.layer_id == plane))
        for plane in (300, 310, 320)
    ]
    assert innermost_apothems == pytest.approx([520.0, 620.0, 720.0])


def test_a_ray_from_the_origin_crosses_each_plane_exactly_once():
    """A muon leaving the interaction point crosses one side per plane, so the
    3 stations x 3 planes give it exactly 9 hits -- the triplet structure only
    works because the polygon is convex around the origin."""
    layers = build_muon_system(
        layer_id_base=300, apothem_inner=520.0, station_spacing=100.0, n_stations=3, n_planes=3
    )
    for phi in (0.0, 0.19, 1.0, -2.4, 3.0):
        trajectory = Trajectory(x0=0.0, y0=0.0, phi0=phi)
        crossed = [layer for layer in layers if first_intersection(trajectory, layer) is not None]
        assert len(crossed) == 9
        assert sorted(layer.layer_id for layer in crossed) == [
            300, 301, 302, 310, 311, 312, 320, 321, 322
        ]


def test_a_ray_aimed_at_a_vertex_clips_both_sides_meeting_there():
    """The polygon's analogue of a barrel layer's module overlap: at a corner
    the two chambers meet, so a track threading the vertex is seen twice by
    that plane. Real, and worth pinning so it is never mistaken for a bug."""
    sides = build_polygon(layer_id=300, apothem=520.0, n_sides=8)
    vertex_phi = math.pi / 8  # halfway between two side midpoints == a vertex
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=vertex_phi)
    crossed = [side for side in sides if first_intersection(trajectory, side) is not None]
    assert len(crossed) == 2
    # both "hits" are the same physical point -- the shared corner
    hits = [first_intersection(trajectory, side) for side in crossed]
    assert hits[0].x == pytest.approx(hits[1].x)
    assert hits[0].y == pytest.approx(hits[1].y)


def test_phi_offset_rotates_the_whole_polygon():
    plain = build_polygon(layer_id=0, apothem=500.0, n_sides=8)
    rotated = build_polygon(layer_id=0, apothem=500.0, n_sides=8, phi_offset=math.pi / 8)
    assert not _is_parallel(plain[0], rotated[0])
    # rotating by a full side spacing maps the polygon onto itself
    full_step = build_polygon(layer_id=0, apothem=500.0, n_sides=8, phi_offset=2 * math.pi / 8)
    assert _is_parallel(full_step[0], plain[1])


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        polygon_vertices(apothem=500.0, n_sides=2)
    with pytest.raises(ValueError):
        polygon_vertices(apothem=-1.0, n_sides=8)
    with pytest.raises(ValueError, match="station_id_step"):
        build_muon_system(300, 520.0, 100.0, n_stations=3, n_planes=3, station_id_step=3)
