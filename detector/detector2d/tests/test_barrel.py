import math

import pytest

from detector2d.barrel import (
    build_barrel_circle,
    build_barrel_modules,
    module_reach,
    n_modules_for_overlap,
)
from detector2d.geometry import CircleLayer, Trajectory
from detector2d.intersect import intersect


def test_module_reach_at_zero_tilt_is_plain_right_triangle():
    # tilt=0: p1=(R,0), p2=(R,L) -- a right triangle, reach = atan2(L, R).
    reach = module_reach(radius=29.0, length=8.0, tilt=0.0)
    assert reach == pytest.approx(math.atan2(8.0, 29.0))


def test_module_reach_grows_with_tilt():
    reaches = [module_reach(radius=29.0, length=8.0, tilt=math.radians(t)) for t in (0.0, 5.0, 10.0, 15.0)]
    assert reaches == sorted(reaches)
    assert reaches[0] < reaches[-1]


def test_module_reach_rejects_excessive_tilt():
    # tilt > 90deg flips the far tip behind the anchor (negative reach) --
    # invalid geometry, not just a large-but-valid angle.
    with pytest.raises(ValueError):
        module_reach(radius=29.0, length=8.0, tilt=math.radians(100.0))


def test_zero_overlap_is_edge_to_edge_tiling():
    reach = module_reach(radius=29.0, length=8.0, tilt=math.radians(10.0))
    n = n_modules_for_overlap(radius=29.0, length=8.0, tilt=math.radians(10.0), overlap_fraction=0.0)
    assert n == round(2.0 * math.pi / reach)


@pytest.mark.parametrize("overlap_fraction", [0.05, 0.15, 0.30])
def test_n_modules_for_overlap_matches_target_up_to_rounding(overlap_fraction):
    radius, length, tilt = 68.0, 8.0, math.radians(10.0)
    reach = module_reach(radius, length, tilt)
    n = n_modules_for_overlap(radius, length, tilt, overlap_fraction)
    delta = 2.0 * math.pi / n
    actual_overlap = (reach - delta) / reach
    assert actual_overlap == pytest.approx(overlap_fraction, abs=0.03)


def test_n_modules_for_overlap_rejects_out_of_range_fraction():
    with pytest.raises(ValueError):
        n_modules_for_overlap(radius=29.0, length=8.0, tilt=0.1, overlap_fraction=1.0)
    with pytest.raises(ValueError):
        n_modules_for_overlap(radius=29.0, length=8.0, tilt=0.1, overlap_fraction=-0.1)


def test_build_barrel_modules_shares_layer_id_and_pitch():
    modules = build_barrel_modules(
        layer_id=3, radius=29.0, half_length=4.0, tilt=math.radians(10.0), overlap_fraction=0.15, pitch=0.1
    )
    n = n_modules_for_overlap(radius=29.0, length=8.0, tilt=math.radians(10.0), overlap_fraction=0.15)
    assert len(modules) == n
    assert all(m.layer_id == 3 for m in modules)
    assert all(m.pitch == 0.1 for m in modules)
    # every module's proximal edge (p1) is anchored exactly on the circle
    for m in modules:
        assert math.hypot(*m.p1) == pytest.approx(29.0)


def test_build_barrel_modules_covers_full_circle():
    modules = build_barrel_modules(
        layer_id=0, radius=100.0, half_length=8.0, tilt=math.radians(8.0), overlap_fraction=0.10
    )
    anchors = sorted(math.atan2(m.p1[1], m.p1[0]) for m in modules)
    n = len(anchors)
    deltas = [(anchors[i + 1] - anchors[i]) for i in range(n - 1)] + [(anchors[0] - anchors[-1]) % (2 * math.pi)]
    # evenly spaced anchors around the full circle
    for d in deltas:
        assert d == pytest.approx(2 * math.pi / n)


def test_build_barrel_circle_matches_plain_circle_layer():
    layer = build_barrel_circle(layer_id=5, radius=48.0, pitch=0.5)
    assert layer == CircleLayer(layer_id=5, center=(0.0, 0.0), radius=48.0, pitch=0.5)


def test_overlap_zone_produces_a_double_hit():
    """A straight track aimed into the angular overlap between module 0 and
    module 1 must cross both -- the double-hit behavior the tilt/overlap
    machinery exists for."""
    radius, half_length, tilt_deg, overlap_fraction = 29.0, 4.0, 10.0, 0.15
    tilt = math.radians(tilt_deg)
    modules = build_barrel_modules(
        layer_id=0, radius=radius, half_length=half_length, tilt=tilt, overlap_fraction=overlap_fraction
    )
    n = len(modules)
    delta = 2.0 * math.pi / n
    reach = module_reach(radius, 2 * half_length, tilt)
    assert reach > delta  # a real overlap exists to aim into

    phi_in_overlap = (delta + reach) / 2.0
    traj = Trajectory(x0=0.0, y0=0.0, phi0=phi_in_overlap)
    hits = [h for m in modules for h in intersect(traj, m)]
    assert len(hits) == 2

    # squarely inside a single module's own (non-overlapping) exclusive zone:
    # back to a single hit. Checked for module 0 and, by the same offset one
    # spacing around, for module 1.
    for phi in (delta * 0.5, 1.5 * delta):
        traj = Trajectory(x0=0.0, y0=0.0, phi0=phi)
        hits = [h for m in modules for h in intersect(traj, m)]
        assert len(hits) == 1
