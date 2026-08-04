"""Core 2D geometry: detector layers and charged-particle trajectories.

Our world only has x and y. A 3D "plane" detector becomes a straight line
segment (:class:`LineLayer`); a 3D barrel "cylinder" becomes a circle
(:class:`CircleLayer`). A charged particle in a constant field pointing out
of the plane (Bz) moves on a circular arc; with no field (or a neutral
particle) it moves on a straight line, which is the R -> infinity limit of
the same arc -- both are represented by :class:`Trajectory`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LineLayer:
    """A straight detector layer (2D analogue of a 3D plane): the segment p1->p2.

    ``pitch``, if set, is the segmentation cell size along the line (distance
    from ``p1``), used by the clustering package to digitize hits into cells.
    ``system`` names the subsystem this layer belongs to (see
    :class:`CircleLayer`).
    """

    layer_id: int
    p1: tuple[float, float]
    p2: tuple[float, float]
    pitch: float | None = None
    system: str = "tracker"

    @property
    def direction(self) -> tuple[float, float]:
        (x1, y1), (x2, y2) = self.p1, self.p2
        return (x2 - x1, y2 - y1)

    @property
    def length(self) -> float:
        dx, dy = self.direction
        return math.hypot(dx, dy)


@dataclass(frozen=True)
class CircleLayer:
    """A circular detector layer (2D analogue of a 3D barrel cylinder).

    ``pitch``, if set, is the segmentation cell size (arc length) along the
    circle, used by the clustering package to digitize hits into cells.

    ``system`` names the subsystem this layer belongs to -- ``"tracker"``
    (the default, so every pre-existing layout keeps its meaning),
    ``"ecal"``, ``"hcal"``, ``"muon"``. Simulation code dispatches on it (a
    calorimeter layer collects energy, a tracker/muon layer records a
    position hit) rather than on ``layer_id`` ranges.
    """

    layer_id: int
    center: tuple[float, float]
    radius: float
    pitch: float | None = None
    system: str = "tracker"


@dataclass(frozen=True)
class Trajectory:
    """A charged particle's path starting at ``(x0, y0)`` heading at angle ``phi0``.

    ``radius=None`` (or an infinite radius) is a straight track. A finite
    signed radius is a circular arc of that (geometric) radius: positive
    curls left (counter-clockwise) as the particle moves forward, negative
    curls right. This sign convention is exactly ``1/kappa`` for the usual
    signed track curvature ``kappa = q*Bz/pt`` (see :mod:`detector2d.field`).
    """

    x0: float
    y0: float
    phi0: float
    radius: float | None = None

    @property
    def is_straight(self) -> bool:
        return self.radius is None or math.isinf(self.radius)

    @property
    def center(self) -> tuple[float, float]:
        """Center of the circular arc. Undefined for a straight trajectory."""
        if self.is_straight:
            raise ValueError("a straight trajectory has no center")
        r = self.radius
        return (self.x0 - r * math.sin(self.phi0), self.y0 + r * math.cos(self.phi0))

    def position(self, s: float) -> tuple[float, float]:
        """Position at arc length ``s`` along the trajectory (``s >= 0`` is forward)."""
        if self.is_straight:
            return (self.x0 + s * math.cos(self.phi0), self.y0 + s * math.sin(self.phi0))
        r = self.radius
        cx, cy = self.center
        phi = self.phi0 + s / r
        return (cx + r * math.sin(phi), cy - r * math.cos(phi))

    def direction_at(self, s: float) -> float:
        """Direction angle (radians) of travel at arc length ``s``."""
        if self.is_straight:
            return self.phi0
        return self.phi0 + s / self.radius

    @property
    def d0(self) -> float:
        """Signed transverse impact parameter: the trajectory's closest
        (signed) distance to the origin -- the usual tracking-detector
        "impact parameter" observable, and what a displaced (b-jet-like)
        vertex shows up as.

        For a straight trajectory this is the perpendicular distance from
        the origin to the line through ``(x0, y0)`` along ``phi0``: positive
        if the origin is to the trajectory's left (in its direction of
        travel), negative if to its right. A circular arc's value --
        ``sign(radius) * (|radius| - |center distance to origin|)`` -- is
        defined so it agrees with the straight-line formula in the
        ``radius -> +-inf`` limit, i.e. it is continuous across
        ``is_straight``.
        """
        if self.is_straight:
            return self.x0 * math.sin(self.phi0) - self.y0 * math.cos(self.phi0)
        cx, cy = self.center
        sign = math.copysign(1.0, self.radius)
        return sign * (abs(self.radius) - math.hypot(cx, cy))
