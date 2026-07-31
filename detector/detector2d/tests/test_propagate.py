import math

import pytest
from detector2d.field import FieldRegion, FieldRegions, signed_radius
from detector2d.geometry import CircleLayer, LineLayer, Trajectory
from detector2d.propagate import SegmentedTrajectory, intersect_segmented, propagate

K = 0.2998


def _three_region_field(bz_inner=2.0, bz_mid=0.0, bz_outer=-1.0):
    """The real layout: strong in the tracker, none in the calorimeters,
    half and reversed in the muon system's flux return."""
    return FieldRegions(
        regions=(
            FieldRegion(r_max=210.0, bz=bz_inner),
            FieldRegion(r_max=480.0, bz=bz_mid),
            FieldRegion(r_max=None, bz=bz_outer),
        ),
        k=K,
    )


# --- FieldRegions ------------------------------------------------------------


def test_region_lookup_by_radius():
    field = _three_region_field()
    assert field.bz_at(0.0) == 2.0
    assert field.bz_at(209.9) == 2.0
    assert field.bz_at(210.0) == 0.0  # a radius on a boundary belongs to the outer region
    assert field.bz_at(479.9) == 0.0
    assert field.bz_at(1000.0) == -1.0
    # the outermost region is unbounded, so the field map itself has no outer edge
    assert field.outer_radius is None
    assert FieldRegions(regions=(FieldRegion(210.0, 2.0), FieldRegion(480.0, 0.0))).outer_radius == 480.0


def test_region_bounds():
    field = _three_region_field()
    assert field.region_bounds(0) == (0.0, 210.0)
    assert field.region_bounds(1) == (210.0, 480.0)
    assert field.region_bounds(2) == (480.0, None)


def test_malformed_region_stacks_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        FieldRegions(regions=())
    with pytest.raises(ValueError, match="strictly increase"):
        FieldRegions(regions=(FieldRegion(200.0, 1.0), FieldRegion(100.0, 0.0)))
    with pytest.raises(ValueError, match="outermost"):
        FieldRegions(regions=(FieldRegion(None, 1.0), FieldRegion(300.0, 0.0)))


# --- propagate: the degenerate single-region case ----------------------------


def test_single_region_reproduces_a_plain_trajectory_point_for_point():
    """The generalization must be exact, not approximate: with one region the
    segmented path *is* the arc the old code would have produced."""
    field = FieldRegions.constant(bz=1.0, k=K)
    pt, charge = 50.0, 1.0
    path = propagate(0.0, 0.0, 0.3, charge, pt, field, max_path_length=300.0)
    plain = Trajectory(x0=0.0, y0=0.0, phi0=0.3, radius=signed_radius(pt, charge, 1.0, K))

    assert len(path.segments) == 1
    for s in (0.0, 10.0, 137.5, 300.0):
        assert path.position(s) == pytest.approx(plain.position(s))
        assert path.direction_at(s) == pytest.approx(plain.direction_at(s))


def test_neutral_particle_is_straight_in_every_region():
    field = _three_region_field()
    path = propagate(0.0, 0.0, 0.4, charge=0.0, pt=30.0, field=field, world_radius=800.0)
    assert path.is_straight
    x, y = path.position(500.0)
    assert (x, y) == pytest.approx((500.0 * math.cos(0.4), 500.0 * math.sin(0.4)))


# --- propagate: crossing regions ---------------------------------------------


def test_path_is_continuous_in_position_and_direction_across_boundaries():
    field = _three_region_field()
    path = propagate(0.0, 0.0, 0.2, charge=1.0, pt=300.0, field=field, world_radius=800.0)
    assert len(path.segments) >= 3

    for segment in path.segments[1:]:
        s = segment.s_start
        before_xy, after_xy = path.position(s - 1e-6), path.position(s + 1e-6)
        assert before_xy == pytest.approx(after_xy, abs=1e-4)
        before_phi, after_phi = path.direction_at(s - 1e-6), path.direction_at(s + 1e-6)
        assert math.remainder(after_phi - before_phi, 2 * math.pi) == pytest.approx(0.0, abs=1e-4)


def test_segment_boundaries_land_on_the_field_region_boundaries():
    field = _three_region_field()
    path = propagate(0.0, 0.0, 0.2, charge=1.0, pt=300.0, field=field, world_radius=800.0)
    crossing_radii = [math.hypot(*path.position(seg.s_start)) for seg in path.segments[1:]]
    assert crossing_radii[0] == pytest.approx(210.0, abs=1e-3)
    assert crossing_radii[1] == pytest.approx(480.0, abs=1e-3)


def test_zero_field_region_gives_a_straight_segment():
    field = _three_region_field()
    path = propagate(0.0, 0.0, 0.2, charge=1.0, pt=300.0, field=field, world_radius=800.0)
    assert not path.segments[0].trajectory.is_straight  # tracker: bends
    assert path.segments[1].trajectory.is_straight  # calorimeters: bz = 0
    assert not path.segments[2].trajectory.is_straight  # muon system: bends again


def test_reversed_outer_field_bends_the_track_the_other_way():
    """The muon system's reflux is the whole point: a positive particle that
    curls one way in the tracker must curl the *opposite* way outside it."""
    field = _three_region_field(bz_inner=2.0, bz_mid=0.0, bz_outer=-1.0)
    path = propagate(0.0, 0.0, 0.2, charge=1.0, pt=300.0, field=field, world_radius=800.0)
    tracker_radius = path.segments[0].trajectory.radius
    muon_radius = path.segments[2].trajectory.radius
    assert tracker_radius * muon_radius < 0  # opposite signed curvature
    # half the field strength -> twice the bend radius
    assert abs(muon_radius) == pytest.approx(2 * abs(tracker_radius))


def test_turning_direction_actually_reverses_along_the_path():
    """Not just the sign of a stored radius -- the heading really turns one
    way inside the tracker and the other way in the muon system."""
    field = _three_region_field()
    path = propagate(0.0, 0.0, 0.2, charge=1.0, pt=300.0, field=field, world_radius=800.0)
    tracker, muon = path.segments[0], path.segments[2]

    def turn(segment):
        span = min(segment.length, 50.0)
        start = segment.trajectory.direction_at(0.0)
        end = segment.trajectory.direction_at(span)
        return math.remainder(end - start, 2 * math.pi)

    assert turn(tracker) * turn(muon) < 0


# --- propagate: stopping conditions ------------------------------------------


def test_propagation_stops_at_the_world_radius():
    field = _three_region_field()
    path = propagate(0.0, 0.0, 0.0, charge=0.0, pt=10.0, field=field, world_radius=800.0)
    assert path.total_length == pytest.approx(800.0)
    assert math.hypot(*path.position(path.total_length)) == pytest.approx(800.0)


def test_a_low_pt_curler_terminates_at_max_path_length_instead_of_looping():
    """A track whose bend radius is too small to leave the tracker would circle
    forever; max_path_length is what makes that finite."""
    field = _three_region_field()
    path = propagate(
        0.0, 0.0, 0.0, charge=1.0, pt=10.0, field=field, world_radius=800.0, max_path_length=500.0
    )
    assert path.total_length == pytest.approx(500.0)
    assert math.hypot(*path.position(path.total_length)) < 210.0  # never escaped the tracker


def test_unbounded_propagation_ends_in_an_infinite_final_segment():
    field = FieldRegions.constant(bz=0.0, k=K)
    path = propagate(0.0, 0.0, 0.0, charge=1.0, pt=10.0, field=field)
    assert math.isinf(path.total_length)
    assert path.position(1e6) == pytest.approx((1e6, 0.0))


def test_max_segments_caps_the_work():
    field = _three_region_field()
    path = propagate(
        0.0, 0.0, 0.0, charge=1.0, pt=100.0, field=field, world_radius=1e9, max_segments=2
    )
    assert len(path.segments) == 2


# --- intersect_segmented -----------------------------------------------------


def test_hit_in_a_later_region_reports_a_global_arc_length():
    """A hit found on the second segment must be reported at its distance from
    the *start of the path*, not from the start of its own segment."""
    field = _three_region_field()
    path = propagate(0.0, 0.0, 0.0, charge=0.0, pt=10.0, field=field, world_radius=800.0)
    layer = CircleLayer(layer_id=7, center=(0.0, 0.0), radius=600.0)
    hits = intersect_segmented(path, layer)
    assert len(hits) == 1
    assert hits[0].s == pytest.approx(600.0)  # straight from the origin
    assert (hits[0].x, hits[0].y) == pytest.approx((600.0, 0.0))
    assert hits[0].s > path.segments[-1].s_start  # genuinely on the last segment


def test_crossings_beyond_a_segments_own_end_are_discarded():
    """Each segment's arc is only physical inside its own region -- extending
    it past the boundary would put hits where the particle never went."""
    inner_only = Trajectory(x0=0.0, y0=0.0, phi0=0.0)
    path = SegmentedTrajectory.single(inner_only, length=100.0)
    reachable = CircleLayer(layer_id=0, center=(0.0, 0.0), radius=50.0)
    beyond = CircleLayer(layer_id=1, center=(0.0, 0.0), radius=150.0)
    assert len(intersect_segmented(path, reachable)) == 1
    assert intersect_segmented(path, beyond) == []


def test_hits_are_returned_sorted_by_global_arc_length():
    field = _three_region_field()
    path = propagate(0.0, 0.0, 0.0, charge=0.0, pt=10.0, field=field, world_radius=800.0)
    layers = [
        LineLayer(layer_id=i, p1=(r, -50.0), p2=(r, 50.0)) for i, r in enumerate((600.0, 100.0, 300.0))
    ]
    hits = [intersect_segmented(path, layer)[0] for layer in layers]
    assert [round(h.s) for h in hits] == [600, 100, 300]
    assert sorted(h.s for h in hits) == [pytest.approx(100.0), pytest.approx(300.0), pytest.approx(600.0)]


def test_segmented_trajectory_needs_at_least_one_segment():
    with pytest.raises(ValueError):
        SegmentedTrajectory(segments=())
