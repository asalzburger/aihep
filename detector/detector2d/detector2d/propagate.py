"""Propagate a particle through a piecewise-constant field: a *segmented*
trajectory.

:class:`~detector2d.geometry.Trajectory` is one arc, because one constant
field gives one constant bend radius. A real detector's field changes with
radius (strong in the solenoid, ~0 in the calorimeters, reversed in the muon
system's flux return), so the path becomes a *chain* of arcs: each shell of
:class:`~detector2d.field.FieldRegions` contributes one arc, and consecutive
arcs join with continuous position and direction.

:class:`SegmentedTrajectory` presents that chain behind the *same* interface a
single ``Trajectory`` has -- ``position(s)``, ``direction_at(s)``, with ``s``
the global arc length from the start -- so drawing code and hit-finding code
consume either without branching. :func:`intersect_path` is the corresponding
generalization of :func:`detector2d.intersect.intersect`.

The single-region case is a strict pass-through: one segment, geometrically
identical to the plain ``Trajectory`` it would have been.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .field import FieldRegions, signed_radius
from .geometry import CircleLayer, Trajectory
from .intersect import Hit, Layer, first_intersection, intersect

_EPS = 1e-9

#: How far past a region boundary to step when starting the next segment.
#: Large enough that the new segment starts unambiguously inside the new
#: region (so re-finding the boundary we just crossed is impossible), small
#: enough to be physically irrelevant at any sane detector scale.
_BOUNDARY_NUDGE = 1e-7


@dataclass(frozen=True)
class Segment:
    """One arc of a :class:`SegmentedTrajectory`, valid for global arc lengths
    ``s_start <= s <= s_start + length``. ``length`` may be ``math.inf`` for a
    final segment that never leaves its region."""

    trajectory: Trajectory
    s_start: float
    length: float

    @property
    def s_end(self) -> float:
        return self.s_start + self.length


@dataclass(frozen=True)
class SegmentedTrajectory:
    """A chain of arcs joined at field-region boundaries.

    Duck-type compatible with :class:`~detector2d.geometry.Trajectory` for
    everything that matters downstream: ``position(s)``, ``direction_at(s)``,
    ``is_straight``. ``s`` is always the *global* arc length from the start of
    the first segment.
    """

    segments: tuple[Segment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.segments:
            raise ValueError("a SegmentedTrajectory needs at least one segment")

    @classmethod
    def single(cls, trajectory: Trajectory, length: float = math.inf) -> SegmentedTrajectory:
        """Wrap one plain ``Trajectory`` -- the degenerate, constant-field case."""
        return cls(segments=(Segment(trajectory=trajectory, s_start=0.0, length=length),))

    @property
    def total_length(self) -> float:
        return self.segments[-1].s_end

    @property
    def is_straight(self) -> bool:
        return all(segment.trajectory.is_straight for segment in self.segments)

    @property
    def x0(self) -> float:
        return self.segments[0].trajectory.x0

    @property
    def y0(self) -> float:
        return self.segments[0].trajectory.y0

    @property
    def phi0(self) -> float:
        return self.segments[0].trajectory.phi0

    def segment_at(self, s: float) -> Segment:
        """The segment covering global arc length ``s``, clamped to the first
        or last segment for ``s`` outside the path (so ``position``
        extrapolates along the end arcs rather than raising)."""
        for segment in self.segments:
            if s <= segment.s_end + _EPS:
                return segment
        return self.segments[-1]

    def position(self, s: float) -> tuple[float, float]:
        segment = self.segment_at(s)
        return segment.trajectory.position(s - segment.s_start)

    def direction_at(self, s: float) -> float:
        segment = self.segment_at(s)
        return segment.trajectory.direction_at(s - segment.s_start)


def _boundary_crossing(trajectory: Trajectory, radius: float | None) -> float | None:
    """Smallest positive arc length at which ``trajectory`` crosses the circle
    of ``radius`` about the origin, or ``None``. Just
    :func:`detector2d.intersect.first_intersection` against a bare
    ``CircleLayer`` -- the boundary is only another circular layer."""
    if radius is None:
        return None
    hit = first_intersection(trajectory, CircleLayer(layer_id=-1, center=(0.0, 0.0), radius=radius))
    return hit.s if hit is not None else None


def propagate(
    x0: float,
    y0: float,
    phi0: float,
    charge: float,
    pt: float,
    field: FieldRegions,
    world_radius: float | None = None,
    max_path_length: float | None = None,
    max_segments: int = 64,
) -> SegmentedTrajectory:
    """Follow a particle of ``pt``/``charge`` from ``(x0, y0)`` heading ``phi0``
    through ``field``, producing one arc per field region it traverses.

    Propagation stops at whichever comes first: leaving ``world_radius``,
    reaching ``max_path_length``, or ``max_segments`` region crossings. With
    no stopping condition at all, a particle that never leaves its region
    (a neutral going straight in an unbounded field map, or a low-``pt``
    curler that loops forever) ends in a final segment of infinite length --
    finite work, unbounded reach.

    ``max_path_length`` is what a caller uses to stop that curler: without it,
    a track whose bend radius is too small to escape the tracker keeps
    circling and keeps re-crossing layers it has already left.
    """
    limit = math.inf if max_path_length is None else max_path_length
    segments: list[Segment] = []

    x, y, phi = x0, y0, phi0
    probe_r = math.hypot(x0, y0)
    s_global = 0.0

    for _ in range(max_segments):
        region_index = field.region_index(probe_r)
        r_inner, r_outer = field.region_bounds(region_index)
        radius = signed_radius(pt, charge, field.regions[region_index].bz, field.k)
        trajectory = Trajectory(x0=x, y0=y, phi0=phi, radius=None if math.isinf(radius) else radius)

        # Candidate stopping points: this region's own two boundaries...
        candidates = [
            s
            for s in (
                _boundary_crossing(trajectory, r_inner if r_inner > 0.0 else None),
                _boundary_crossing(trajectory, r_outer),
            )
            if s is not None and s > _BOUNDARY_NUDGE
        ]
        step = min(candidates) if candidates else math.inf

        # ...and the two hard stops, which end propagation rather than
        # starting another segment.
        world_s = _boundary_crossing(trajectory, world_radius)
        remaining = limit - s_global
        stop = min(remaining, world_s if world_s is not None else math.inf)

        if step >= stop:
            segments.append(Segment(trajectory=trajectory, s_start=s_global, length=stop))
            return SegmentedTrajectory(segments=tuple(segments))

        # Cross into the next region, nudging just past the boundary so the
        # next segment cannot re-discover the boundary it just left.
        advance = step + _BOUNDARY_NUDGE
        segments.append(Segment(trajectory=trajectory, s_start=s_global, length=advance))
        x, y = trajectory.position(advance)
        phi = trajectory.direction_at(advance)
        probe_r = math.hypot(x, y)
        s_global += advance

    return SegmentedTrajectory(segments=tuple(segments))


def intersect_segmented(path: SegmentedTrajectory, layer: Layer) -> list[Hit]:
    """Every crossing of ``layer`` along ``path``, sorted by global arc length.

    Each segment is intersected with its own plain ``Trajectory`` (reusing
    :func:`detector2d.intersect.intersect` unchanged); crossings past the
    segment's own end belong to a region the particle is no longer in and are
    discarded, and the survivors' ``s`` is shifted to the global arc length.
    """
    hits: list[Hit] = []
    for segment in path.segments:
        for hit in intersect(segment.trajectory, layer):
            if hit.s > segment.length + _EPS:
                continue
            hits.append(
                Hit(s=segment.s_start + hit.s, x=hit.x, y=hit.y, local_coord=hit.local_coord)
            )
    return sorted(hits, key=lambda hit: hit.s)


def intersect_path(path: SegmentedTrajectory | Trajectory, layer: Layer) -> list[Hit]:
    """:func:`intersect` or :func:`intersect_segmented`, whichever fits -- so
    callers holding "a path" never have to branch on which kind it is."""
    if isinstance(path, SegmentedTrajectory):
        return intersect_segmented(path, layer)
    return intersect(path, layer)


def first_intersection_path(path: SegmentedTrajectory | Trajectory, layer: Layer) -> Hit | None:
    """The earliest crossing of ``layer`` along ``path``, or ``None``."""
    hits = intersect_path(path, layer)
    return hits[0] if hits else None
