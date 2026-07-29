"""Barrel (cylindrical) layer construction: approximate a circular layer by a
ring of identical, tilted planar sensor modules (2D analogue of a real
tracker's tilted/shingled barrel layer), with a controlled azimuthal overlap
between neighbors so a single track can cross two adjacent modules and
produce a genuine double hit on one physical layer.

Each module is anchored by its own proximal edge exactly on the circle of
``radius``, at evenly spaced angles, and extends a fixed length tangentially
with a small inward tilt (mixing in the radial direction). Tilting a chord
symmetrically about its own center only *shortens* its azimuthal footprint
(a plain cosine foreshortening) -- anchoring one edge and tilting the
extension is what lets identical, evenly-spaced modules cover more azimuth
than their center-to-center spacing, which is the overlap we want.

Given the module's own angular reach (anchor to far tip) at the requested
tilt, the module count ``N`` is solved in closed form so that packing them
at that spacing reproduces the requested ``overlap_fraction`` exactly (up to
rounding ``N`` to the nearest integer) -- no iterative solve needed.
"""

from __future__ import annotations

import math

from .geometry import CircleLayer, LineLayer

_EPS = 1e-9


def build_barrel_circle(layer_id: int, radius: float, pitch: float | None = None) -> CircleLayer:
    """The simplified-mode barrel layer: just the bare circle."""
    return CircleLayer(layer_id=layer_id, center=(0.0, 0.0), radius=radius, pitch=pitch)


def _module_endpoints(
    radius: float, length: float, theta: float, tilt: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """A module's endpoints: proximal edge ``p1`` anchored on the circle at
    angle ``theta``, extending ``length`` in the tangent direction tilted by
    ``tilt`` toward the center (mixing in ``-radial``)."""
    tangent = (-math.sin(theta), math.cos(theta))
    radial = (math.cos(theta), math.sin(theta))
    ux = math.cos(tilt) * tangent[0] - math.sin(tilt) * radial[0]
    uy = math.cos(tilt) * tangent[1] - math.sin(tilt) * radial[1]
    p1 = (radius * math.cos(theta), radius * math.sin(theta))
    p2 = (p1[0] + length * ux, p1[1] + length * uy)
    return p1, p2


def module_reach(radius: float, length: float, tilt: float) -> float:
    """Angular distance (radians, ``> 0``) from a module's own anchor to its
    far tip, for a module of full ``length`` anchored on ``radius`` and
    tilted by ``tilt``. This is the module's azimuthal footprint, used both
    to solve the module count for a target overlap and to place modules."""
    _, p2 = _module_endpoints(radius, length, theta=0.0, tilt=tilt)
    reach = math.atan2(p2[1], p2[0])
    if not (_EPS < reach < math.pi - _EPS):
        raise ValueError(
            f"module reach {reach!r} rad is out of the valid (0, pi) range for "
            f"radius={radius!r}, length={length!r}, tilt={tilt!r} -- module too "
            "long, or tilt too large, for this radius"
        )
    return reach


def n_modules_for_overlap(radius: float, length: float, tilt: float, overlap_fraction: float) -> int:
    """Number of evenly-spaced modules (anchors at spacing ``2*pi/N``) whose
    own angular ``module_reach`` overlaps each neighbor's territory by
    ``overlap_fraction`` of that reach."""
    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError(f"overlap_fraction must be in [0, 1), got {overlap_fraction!r}")
    reach = module_reach(radius, length, tilt)
    spacing = reach * (1.0 - overlap_fraction)
    n = round(2.0 * math.pi / spacing)
    if n < 3:
        raise ValueError(f"resolved module count {n} is too small (radius/length/tilt too coarse)")
    return n


def build_barrel_modules(
    layer_id: int,
    radius: float,
    half_length: float,
    tilt: float,
    overlap_fraction: float,
    pitch: float | None = None,
) -> list[LineLayer]:
    """The detailed-mode barrel layer: a ring of identical tilted `LineLayer`
    modules approximating a circle of `radius`, all tagged with `layer_id`.

    `half_length` is half the module's physical length (full length
    `2*half_length`), the same for every module in this layer. `tilt` (in
    radians) tips each module's far end toward the center relative to pure
    tangential; `overlap_fraction` is the target azimuthal overlap between
    neighboring modules (see module docstring). The module count is derived,
    not configured -- see `n_modules_for_overlap`.
    """
    length = 2.0 * half_length
    n = n_modules_for_overlap(radius, length, tilt, overlap_fraction)
    delta = 2.0 * math.pi / n
    modules = []
    for i in range(n):
        theta = i * delta
        p1, p2 = _module_endpoints(radius, length, theta, tilt)
        modules.append(LineLayer(layer_id=layer_id, p1=p1, p2=p2, pitch=pitch))
    return modules
