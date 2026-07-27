import math

import pytest

from detector2d.geometry import CircleLayer, LineLayer, Trajectory


def test_straight_trajectory_position():
    traj = Trajectory(x0=1.0, y0=2.0, phi0=math.pi / 4)
    assert traj.is_straight
    x, y = traj.position(math.sqrt(2))
    assert x == pytest.approx(2.0)
    assert y == pytest.approx(3.0)
    assert traj.direction_at(5.0) == pytest.approx(math.pi / 4)


def test_straight_trajectory_via_infinite_radius():
    straight = Trajectory(x0=0, y0=0, phi0=0.3, radius=None)
    infinite = Trajectory(x0=0, y0=0, phi0=0.3, radius=math.inf)
    assert infinite.is_straight
    assert straight.position(10) == pytest.approx(infinite.position(10))


def test_circular_trajectory_starts_at_vertex():
    traj = Trajectory(x0=5.0, y0=-3.0, phi0=1.234, radius=17.0)
    assert traj.position(0.0) == pytest.approx((5.0, -3.0))
    assert traj.direction_at(0.0) == pytest.approx(1.234)


def test_ccw_quarter_turn_left_curl():
    # heading east (phi0=0), positive radius curls left (CCW): quarter turn
    # should land north-east of the start, directly right of the center.
    traj = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=10.0)
    s_quarter = math.pi * 10.0 / 2.0
    x, y = traj.position(s_quarter)
    assert (x, y) == pytest.approx((10.0, 10.0))
    assert traj.direction_at(s_quarter) == pytest.approx(math.pi / 2)


def test_negative_radius_curls_right():
    traj = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=-10.0)
    s_quarter = math.pi * 10.0 / 2.0
    x, y = traj.position(s_quarter)
    assert (x, y) == pytest.approx((10.0, -10.0))


def test_line_layer_length_and_direction():
    layer = LineLayer(layer_id=0, p1=(0.0, 0.0), p2=(3.0, 4.0))
    assert layer.length == pytest.approx(5.0)
    assert layer.direction == pytest.approx((3.0, 4.0))


def test_circle_layer_is_a_plain_value_holder():
    layer = CircleLayer(layer_id=1, center=(1.0, 1.0), radius=50.0, pitch=2.0)
    assert layer.center == (1.0, 1.0)
    assert layer.radius == 50.0
    assert layer.pitch == 2.0
