"""Turn a physical transverse momentum + charge + field into a signed bend radius.

This is the only place physical units (GeV, Tesla, meters) enter the
package; everything else (:mod:`detector2d.geometry`,
:mod:`detector2d.intersect`) works in an arbitrary length unit and a signed
``radius`` directly. Callers who already know the radius they want (e.g. a
radius fit straight out of a picture, in pixels) can skip this module
entirely and construct a :class:`~detector2d.geometry.Trajectory` directly.

A real detector's field is not one constant everywhere: it is strong inside
the solenoid (the tracker), essentially zero in the calorimeters outside it,
and *reversed* in the muon system where the flux returns. :class:`FieldRegions`
describes that as a radial stack of constant-``bz`` shells;
:mod:`detector2d.propagate` turns it into a multi-segment trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Standard R[m] = pt[GeV] / (k * |q| * B[T]) constant (k = 0.3 * c-derived).
DEFAULT_K = 0.2998


def signed_radius(pt: float, charge: float, bz: float, k: float = DEFAULT_K) -> float:
    """Signed bend radius of a particle with transverse momentum ``pt`` (>=0),
    charge ``charge`` (in units of e, sign included), in a field ``bz`` out of
    the 2D plane. Returns ``math.inf`` for a neutral particle or zero field
    (straight track) -- pass the result straight into ``Trajectory(radius=...)``.
    """
    if charge == 0 or bz == 0:
        return math.inf
    return pt / (k * charge * bz)


@dataclass(frozen=True)
class FieldRegion:
    """A radial shell of constant field, reaching out to ``r_max``.

    ``r_max=None`` means "out to infinity" and is only valid for the last
    region. The inner edge is the previous region's ``r_max`` (0 for the
    first), so a stack of regions is described by its boundaries alone.
    """

    r_max: float | None
    bz: float


@dataclass(frozen=True)
class FieldRegions:
    """A piecewise-constant, purely radial field: concentric shells of
    constant ``bz``, ordered inside-out.

    The single-region case ``FieldRegions([FieldRegion(None, bz)])`` is
    exactly the old constant-field behaviour, which is how
    :mod:`detector2d.propagate` stays a strict generalization rather than a
    replacement.
    """

    regions: tuple[FieldRegion, ...] = field(default_factory=tuple)
    k: float = DEFAULT_K

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))
        if not self.regions:
            raise ValueError("FieldRegions needs at least one region")
        previous = 0.0
        for i, region in enumerate(self.regions):
            if region.r_max is None:
                if i != len(self.regions) - 1:
                    raise ValueError(
                        f"region {i} has an unbounded r_max but is not the outermost of "
                        f"{len(self.regions)} -- only the last region may be unbounded"
                    )
                continue
            if region.r_max <= previous:
                raise ValueError(
                    f"region boundaries must strictly increase outward, got r_max={region.r_max!r} "
                    f"after {previous!r} at region {i}"
                )
            previous = region.r_max

    @classmethod
    def constant(cls, bz: float, k: float = DEFAULT_K) -> FieldRegions:
        """The degenerate single-region case: one constant ``bz`` everywhere."""
        return cls(regions=(FieldRegion(r_max=None, bz=bz),), k=k)

    def region_index(self, r: float) -> int:
        """Index of the region containing radius ``r``. A radius exactly on a
        boundary belongs to the *outer* region, matching each region's
        half-open ``(r_inner, r_max]``-complement convention -- propagation
        nudges across boundaries anyway, so this only fixes the tie."""
        for i, region in enumerate(self.regions):
            if region.r_max is None or r < region.r_max:
                return i
        return len(self.regions) - 1

    def region_bounds(self, index: int) -> tuple[float, float | None]:
        """``(r_inner, r_max)`` of region ``index``; ``r_max`` may be ``None``."""
        inner = 0.0 if index == 0 else self.regions[index - 1].r_max
        return (inner or 0.0, self.regions[index].r_max)

    def bz_at(self, r: float) -> float:
        """The field at radius ``r``."""
        return self.regions[self.region_index(r)].bz

    def radius_at(self, r: float, pt: float, charge: float) -> float:
        """Signed bend radius a particle of ``pt``/``charge`` has at radius ``r``."""
        return signed_radius(pt, charge, self.bz_at(r), self.k)

    @property
    def outer_radius(self) -> float | None:
        """The outermost boundary, or ``None`` if the field is unbounded."""
        return self.regions[-1].r_max
