import math

import pytest

from sensor.sim.geometry import (
    charge_endpoints,
    deposited_charge,
    segment_pixel_fractions,
)


def test_single_pixel_point():
    fracs = segment_pixel_fractions((10, 10), (10, 10), 25, 50, 10, 10)
    assert fracs == {(0, 0): 1.0}


def test_horizontal_segment_four_columns():
    # pitch_x=25 -> columns at [0,25),[25,50),[50,75),[75,100)
    fracs = segment_pixel_fractions((5, 10), (95, 10), 25, 50, 10, 10)
    assert set(fracs) == {(0, 0), (1, 0), (2, 0), (3, 0)}
    assert fracs[(0, 0)] == pytest.approx(20 / 90)
    assert fracs[(1, 0)] == pytest.approx(25 / 90)
    assert fracs[(2, 0)] == pytest.approx(25 / 90)
    assert fracs[(3, 0)] == pytest.approx(20 / 90)
    assert sum(fracs.values()) == pytest.approx(1.0)


def test_diagonal_segment_conserves_total_fraction():
    fracs = segment_pixel_fractions((3, 4), (123, 217), 25, 50, 10, 10)
    assert sum(fracs.values()) == pytest.approx(1.0)


def test_clipped_segment_loses_charge_outside_grid():
    # grid spans x in [0,250), segment runs from -50 to 300: only the middle
    # portion (t in [50/350, 300/350]) can land in-bounds.
    fracs = segment_pixel_fractions((-50, 10), (300, 10), 25, 50, 10, 10)
    in_bounds_fraction = 250 / 350
    assert sum(fracs.values()) == pytest.approx(in_bounds_fraction)


def test_endpoints_no_drift_matches_straight_track():
    p0, p1 = charge_endpoints(x0=10, y0=20, dxdz=0.1, dydz=-0.2, thickness_um=150, lorentz_slope=0.0)
    assert p0 == (10, 20)
    assert p1 == pytest.approx((10 + 15, 20 - 30))


def test_endpoints_with_drift_shifts_entry_point_only():
    p0, p1 = charge_endpoints(x0=10, y0=20, dxdz=0.0, dydz=0.0, thickness_um=150, lorentz_slope=0.05)
    assert p0 == pytest.approx((10 + 150 * 0.05, 20))
    assert p1 == (10, 20)


def test_deposited_charge_scales_with_path_length():
    q_perp = deposited_charge(0.0, 0.0, thickness_um=150, charge_per_um=1 / 150)
    assert q_perp == pytest.approx(1.0)

    q_angled = deposited_charge(0.3, 0.0, thickness_um=150, charge_per_um=1 / 150)
    assert q_angled == pytest.approx(math.sqrt(1 + 0.3**2))
    assert q_angled > q_perp
