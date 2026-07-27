import math

import numpy as np
import pytest

from hopfield_tracking.coefficients import (
    build_weight_matrix,
    circle_fit_chi2,
    classify_pair,
    fit_circle,
    type1_coefficient,
    type2_coefficient,
)
from hopfield_tracking.network import Segment


def _seg(index, start_id, end_id, start_xy, end_xy):
    return Segment(index=index, start_id=start_id, end_id=end_id, start_xy=start_xy, end_xy=end_xy)


def test_classify_pair_reverse():
    a = _seg(0, 0, 1, (0, 0), (1, 0))
    b = _seg(1, 1, 0, (1, 0), (0, 0))
    assert classify_pair(a, b) == "reverse"


def test_classify_pair_chain():
    a = _seg(0, 0, 1, (0, 0), (1, 0))
    b = _seg(1, 1, 2, (1, 0), (2, 0))
    assert classify_pair(a, b) == "chain"
    assert classify_pair(b, a) == "chain"  # symmetric regardless of argument order


def test_classify_pair_compete_same_start():
    a = _seg(0, 0, 1, (0, 0), (1, 0))
    b = _seg(1, 0, 2, (0, 0), (2, 1))
    assert classify_pair(a, b) == "compete"


def test_classify_pair_compete_same_end():
    a = _seg(0, 0, 2, (0, 0), (2, 0))
    b = _seg(1, 1, 2, (1, 1), (2, 0))
    assert classify_pair(a, b) == "compete"


def test_classify_pair_type2_when_no_shared_point():
    a = _seg(0, 0, 1, (0, 0), (1, 0))
    b = _seg(1, 2, 3, (10, 10), (11, 10))
    assert classify_pair(a, b) == "type2"


def test_type1_coefficient_rewards_smooth_continuation():
    # two collinear segments, straight through -> theta=0, strongly positive
    a = _seg(0, 0, 1, (0, 0), (1, 0))
    b = _seg(1, 1, 2, (1, 0), (2, 0))
    assert type1_coefficient(a, b, n=5) > 0
    assert type1_coefficient(a, b, n=5) == pytest.approx(0.5)  # cos(0)^5 / |1+1| = 1/2


def test_type1_coefficient_inhibits_sharp_reversal():
    # b doubles straight back on a -> theta=pi, cos^5(pi) = -1, strongly negative
    a = _seg(0, 0, 1, (0, 0), (1, 0))
    b = _seg(1, 1, 2, (1, 0), (0, 0))
    assert type1_coefficient(a, b, n=5) < 0


def test_type1_coefficient_scales_with_r_scale():
    a = _seg(0, 0, 1, (0, 0), (1, 0))
    b = _seg(1, 1, 2, (1, 0), (2, 0))
    assert type1_coefficient(a, b, n=5, r_scale=2.0) == pytest.approx(2 * type1_coefficient(a, b, n=5, r_scale=1.0))


def test_fit_circle_recovers_known_circle():
    theta = np.linspace(0, 1.5, 6)
    cx, cy, r = 3.0, -2.0, 5.0
    points = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])
    fit_cx, fit_cy, fit_r = fit_circle(points)
    assert (fit_cx, fit_cy, fit_r) == pytest.approx((cx, cy, r))


def test_circle_fit_chi2_near_zero_for_points_on_a_circle():
    theta = np.linspace(0, 2, 6)
    points = np.column_stack([10 * np.cos(theta), 10 * np.sin(theta)])
    assert circle_fit_chi2(points) == pytest.approx(0.0, abs=1e-9)


def test_circle_fit_chi2_large_for_scattered_points():
    points = np.array([[0.0, 0.0], [1.0, 5.0], [-3.0, 2.0], [4.0, -1.0]])
    assert circle_fit_chi2(points) > 0.01


def test_type2_coefficient_zero_when_too_far_apart():
    a = _seg(0, 0, 1, (0, 0), (1, 0))
    b = _seg(1, 2, 3, (1000, 1000), (1001, 1000))
    assert type2_coefficient(a, b, r_c=10.0) == 0.0


def test_build_weight_matrix_is_symmetric_with_zero_diagonal():
    segments = [
        _seg(0, 0, 1, (0, 0), (1, 0)),
        _seg(1, 1, 2, (1, 0), (2, 0)),
        _seg(2, 2, 1, (2, 0), (1, 0)),
    ]
    t = build_weight_matrix(segments, r_c=5.0, r_scale=1.0)
    assert np.allclose(t, t.T)
    assert np.allclose(np.diag(t), 0.0)
