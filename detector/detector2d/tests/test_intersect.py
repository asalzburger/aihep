import math

import pytest

from detector2d.geometry import CircleLayer, LineLayer, Trajectory
from detector2d.intersect import first_intersection, intersect


def test_straight_hits_perpendicular_line():
    traj = Trajectory(x0=0.0, y0=0.0, phi0=0.0)  # heading +x
    layer = LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0))
    hits = intersect(traj, layer)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.s == pytest.approx(10.0)
    assert (hit.x, hit.y) == pytest.approx((10.0, 0.0))
    assert hit.local_coord == pytest.approx(5.0)  # halfway along the 10-long segment


def test_straight_misses_line_outside_segment_bounds():
    traj = Trajectory(x0=0.0, y0=0.0, phi0=0.0)
    layer = LineLayer(layer_id=0, p1=(10.0, 1.0), p2=(10.0, 5.0))  # doesn't cross y=0
    assert intersect(traj, layer) == []


def test_straight_parallel_to_line_never_hits():
    traj = Trajectory(x0=0.0, y0=0.0, phi0=0.0)  # heading +x
    layer = LineLayer(layer_id=0, p1=(-10.0, 3.0), p2=(10.0, 3.0))  # horizontal, parallel
    assert intersect(traj, layer) == []


def test_straight_behind_the_start_is_not_a_hit():
    traj = Trajectory(x0=0.0, y0=0.0, phi0=0.0)  # heading +x
    layer = LineLayer(layer_id=0, p1=(-10.0, -5.0), p2=(-10.0, 5.0))  # behind the start
    assert intersect(traj, layer) == []


def test_straight_through_circle_two_hits():
    traj = Trajectory(x0=-20.0, y0=0.0, phi0=0.0)  # heading +x through the origin
    layer = CircleLayer(layer_id=0, center=(0.0, 0.0), radius=5.0)
    hits = intersect(traj, layer)
    assert len(hits) == 2
    xs = sorted(h.x for h in hits)
    assert xs == pytest.approx([-5.0, 5.0])
    for h in hits:
        assert h.y == pytest.approx(0.0)


def test_straight_tangent_to_circle_is_a_double_hit():
    traj = Trajectory(x0=-20.0, y0=5.0, phi0=0.0)  # grazes the top of the circle
    layer = CircleLayer(layer_id=0, center=(0.0, 0.0), radius=5.0)
    hits = intersect(traj, layer)
    assert len(hits) == 2
    for h in hits:
        assert (h.x, h.y) == pytest.approx((0.0, 5.0))


def test_straight_missing_circle():
    traj = Trajectory(x0=-20.0, y0=100.0, phi0=0.0)
    layer = CircleLayer(layer_id=0, center=(0.0, 0.0), radius=5.0)
    assert intersect(traj, layer) == []


def test_arc_hits_line_at_expected_point_and_arc_length():
    # heading east, positive radius curls left: circle centered (0, 10), r=10.
    traj = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=10.0)
    layer = LineLayer(layer_id=0, p1=(-20.0, 10.0), p2=(20.0, 10.0))  # through the center
    hits = intersect(traj, layer)
    assert len(hits) == 2
    by_x = sorted(hits, key=lambda h: h.x)
    assert by_x[0].x == pytest.approx(-10.0)  # reached only after wrapping most of the way around
    assert by_x[1].x == pytest.approx(10.0)  # reached first, after a quarter turn
    assert by_x[1].s == pytest.approx(math.pi * 10.0 / 2.0)
    assert by_x[1].s < by_x[0].s
    # round-trip: the trajectory really does pass through both reported points at their s
    for h in hits:
        assert traj.position(h.s) == pytest.approx((h.x, h.y), abs=1e-6)


def test_arc_circle_intersection_matches_closed_form():
    # Two unit-ish circles with a known intersection (classic 3-4-5 setup):
    # centered at (0,0) r=5 and (6,0) r=5 intersect at (3, +-4).
    traj = Trajectory(x0=0.0, y0=-5.0, phi0=0.0, radius=5.0)  # circle: center (0,0), r=5
    layer = CircleLayer(layer_id=0, center=(6.0, 0.0), radius=5.0)
    hits = intersect(traj, layer)
    assert len(hits) == 2
    points = sorted((round(h.x, 6), round(h.y, 6)) for h in hits)
    assert points == [(3.0, -4.0), (3.0, 4.0)]
    for h in hits:
        assert traj.position(h.s) == pytest.approx((h.x, h.y), abs=1e-6)


def test_arc_beyond_quarter_turn_is_still_found():
    # A point requiring more than a quarter turn forward must not be dropped
    # just because the naive principal-branch angle wraps to a negative s.
    traj = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=5.0)  # circle: center (0,5), r=5
    layer = LineLayer(layer_id=0, p1=(-5.0, 5.0), p2=(-4.999, 5.0))  # tiny segment at (-5,5)
    hit = first_intersection(traj, layer)
    assert hit is not None
    assert hit.x == pytest.approx(-5.0, abs=1e-3)
    assert hit.s > math.pi * 5.0  # more than half the circumference away


def test_no_intersection_returns_none():
    traj = Trajectory(x0=0.0, y0=100.0, phi0=0.0)
    layer = LineLayer(layer_id=0, p1=(-1.0, -1.0), p2=(1.0, -1.0))
    assert first_intersection(traj, layer) is None


def test_arc_length_matches_each_layers_distance():
    traj = Trajectory(x0=0.0, y0=0.0, phi0=0.0)
    layers_and_expected_s = [
        (LineLayer(layer_id=2, p1=(30.0, -1.0), p2=(30.0, 1.0)), 30.0),
        (LineLayer(layer_id=0, p1=(10.0, -1.0), p2=(10.0, 1.0)), 10.0),
        (LineLayer(layer_id=1, p1=(20.0, -1.0), p2=(20.0, 1.0)), 20.0),
    ]
    for layer, expected_s in layers_and_expected_s:
        assert first_intersection(traj, layer).s == pytest.approx(expected_s)
