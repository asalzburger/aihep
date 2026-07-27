"""Intersect a :class:`~detector2d.geometry.Trajectory` with a detector layer.

Four cases, dispatched on trajectory straightness x layer type:

- straight x line    -> line-line intersection
- straight x circle  -> ray-circle intersection (quadratic)
- arc x line         -> circle-line-segment intersection (quadratic)
- arc x circle       -> circle-circle intersection (radical line)

``local_coord`` on a :class:`Hit` is the position along the layer, in length
units both ways: distance from ``p1`` for a
:class:`~detector2d.geometry.LineLayer`, arc length from the +x axis around
the layer's own center (``radius * angle``) for a
:class:`~detector2d.geometry.CircleLayer`. This is what the clustering
package digitizes into cell indices via each layer's ``pitch``.

A circular trajectory is only inverted to an arc length within one full turn
of its starting angle (the smallest ``|s|`` that reaches the intersection
point) -- adequate for tracks crossing a detector stack without looping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import CircleLayer, LineLayer, Trajectory

_EPS = 1e-9

Layer = LineLayer | CircleLayer


@dataclass(frozen=True)
class Hit:
    s: float  # arc length along the trajectory
    x: float
    y: float
    local_coord: float  # distance along a LineLayer, or angle around a CircleLayer


def intersect(trajectory: Trajectory, layer: Layer) -> list[Hit]:
    """All intersections with ``s > 0``, sorted by increasing ``s``."""
    if isinstance(layer, LineLayer):
        hits = (
            _straight_line(trajectory, layer)
            if trajectory.is_straight
            else _arc_line(trajectory, layer)
        )
    elif isinstance(layer, CircleLayer):
        hits = (
            _straight_circle(trajectory, layer)
            if trajectory.is_straight
            else _arc_circle(trajectory, layer)
        )
    else:
        raise TypeError(f"Unknown layer type: {type(layer)!r}")
    return sorted((h for h in hits if h.s > _EPS), key=lambda h: h.s)


def first_intersection(trajectory: Trajectory, layer: Layer) -> Hit | None:
    """The earliest (smallest positive ``s``) intersection, or ``None``."""
    hits = intersect(trajectory, layer)
    return hits[0] if hits else None


def _line_line_params(
    a: tuple[float, float], u: tuple[float, float], b: tuple[float, float], v: tuple[float, float]
) -> tuple[float, float] | None:
    """Solve ``a + s*u = b + t*v``. Returns ``(s, t)``, or ``None`` if parallel."""
    denom = u[0] * v[1] - u[1] * v[0]
    if abs(denom) < _EPS:
        return None
    dx, dy = b[0] - a[0], b[1] - a[1]
    s = (dx * v[1] - dy * v[0]) / denom
    t = (dx * u[1] - dy * u[0]) / denom
    return s, t


def _arc_length_at_point(trajectory: Trajectory, x: float, y: float) -> float:
    """Smallest positive arc length ``s`` with ``position(s) == (x, y)``, if one
    exists within one extra turn either side of the principal branch (adequate
    for tracks that don't loop more than ~1.5 times before hitting a layer);
    otherwise the principal (possibly negative) value, which the ``s > 0``
    filter in :func:`intersect` will correctly drop."""
    r = trajectory.radius
    cx, cy = trajectory.center
    phi = math.atan2((x - cx) / r, -(y - cy) / r)
    principal = (phi - trajectory.phi0 + math.pi) % (2 * math.pi) - math.pi
    period = 2 * math.pi * r
    candidates = [r * principal, r * principal + period, r * principal - period]
    positive = [s for s in candidates if s > _EPS]
    return min(positive) if positive else r * principal


def _straight_line(trajectory: Trajectory, layer: LineLayer) -> list[Hit]:
    origin = (trajectory.x0, trajectory.y0)
    direction = (math.cos(trajectory.phi0), math.sin(trajectory.phi0))
    result = _line_line_params(origin, direction, layer.p1, layer.direction)
    if result is None:
        return []
    s, t = result
    if not (0.0 <= t <= 1.0):
        return []
    x, y = trajectory.position(s)
    return [Hit(s=s, x=x, y=y, local_coord=t * layer.length)]


def _straight_circle(trajectory: Trajectory, layer: CircleLayer) -> list[Hit]:
    ox, oy = trajectory.x0, trajectory.y0
    dx, dy = math.cos(trajectory.phi0), math.sin(trajectory.phi0)
    cx, cy = layer.center
    fx, fy = ox - cx, oy - cy

    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - layer.radius**2
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return []
    sqrt_disc = math.sqrt(disc)

    hits = []
    for s in ((-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0):
        x, y = trajectory.position(s)
        local_coord = layer.radius * math.atan2(y - cy, x - cx)
        hits.append(Hit(s=s, x=x, y=y, local_coord=local_coord))
    return hits


def _arc_line(trajectory: Trajectory, layer: LineLayer) -> list[Hit]:
    cx, cy = trajectory.center
    r = abs(trajectory.radius)
    x1, y1 = layer.p1
    dx, dy = layer.direction
    fx, fy = x1 - cx, y1 - cy

    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0.0 or a < _EPS:
        return []
    sqrt_disc = math.sqrt(disc)

    hits = []
    for t in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
        if not (0.0 <= t <= 1.0):
            continue
        x, y = x1 + t * dx, y1 + t * dy
        s = _arc_length_at_point(trajectory, x, y)
        hits.append(Hit(s=s, x=x, y=y, local_coord=t * layer.length))
    return hits


def _arc_circle(trajectory: Trajectory, layer: CircleLayer) -> list[Hit]:
    x1, y1 = trajectory.center
    r1 = abs(trajectory.radius)
    x2, y2 = layer.center
    r2 = layer.radius

    d = math.hypot(x2 - x1, y2 - y1)
    if d < _EPS or d > r1 + r2 or d < abs(r1 - r2):
        return []

    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h2 = r1**2 - a**2
    if h2 < 0.0:
        return []
    h = math.sqrt(h2)
    xm, ym = x1 + a * (x2 - x1) / d, y1 + a * (y2 - y1) / d

    hits = []
    for x, y in (
        (xm + h * (y2 - y1) / d, ym - h * (x2 - x1) / d),
        (xm - h * (y2 - y1) / d, ym + h * (x2 - x1) / d),
    ):
        s = _arc_length_at_point(trajectory, x, y)
        local_coord = r2 * math.atan2(y - y2, x - x2)
        hits.append(Hit(s=s, x=x, y=y, local_coord=local_coord))
    return hits
