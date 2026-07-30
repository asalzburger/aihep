"""The "prescription" side of graph building: *how* to decide which pairs of
hits become edges, kept deliberately separate from *what* an edge is
(`graphs.edm`) and *how it's evaluated* (`graphs.build`). Three
prescriptions, growing in specificity:

- `FullyConnected` -- connect every hit to every other hit in the event (the
  trivial baseline; quadratic in hit count, fine for a didactic event).
- `Regional` -- partition each event's hits into azimuthal (phi) sectors and
  only fully-connect within a sector -- cuts combinatorics via pure spatial
  locality, no feature-based rule.
- `ConnectionRules` -- explicit per-feature range gates (`delta_layer_id`,
  `delta_r`, `delta_x`, `delta_phi`), the closest analogue to how real
  doublet/edge construction is done for GNN-based trackers.

`parse_prescription` builds one of these from a plain dict (typically loaded
from YAML by `graphs.config.load_config`) -- the same `kind`-dispatch
pattern as `detector2d.config.parse_layer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

Range = tuple[float, float]


@dataclass(frozen=True)
class FullyConnected:
    """Connect every hit in an event to every other hit. ``directed=True``
    keeps both ``(i, j)`` and ``(j, i)`` as separate edges; otherwise each
    unordered pair produces exactly one edge."""

    directed: bool = False


@dataclass(frozen=True)
class Regional:
    """Partition each event's hits into ``2*pi / phi_width``-wide azimuthal
    sectors (bucketed by ``atan2(y, x)``) and fully-connect only within a
    sector. ``phi_width`` is in radians; hits in different sectors never
    connect, even if they happen to be close in ``(x, y)`` -- a known,
    deliberate simplification (no wraparound stitching across a sector
    boundary)."""

    phi_width: float


@dataclass(frozen=True)
class ConnectionRules:
    """Connect hit ``i -> j`` only if every *given* range contains the
    corresponding signed delta (``dst - src``); a range left as ``None``
    skips that check entirely, so e.g. only ``delta_layer_id`` restricts
    while ``delta_r``/``delta_x``/``delta_phi`` stay unconstrained.
    ``delta_phi`` is wrapped into ``(-pi, pi]`` before the range check,
    since phi is cyclic. Each range is ``(min, max)``, both inclusive; an
    asymmetric range (e.g. ``(1, 3)``) naturally encodes direction (only
    "outward" connects), a symmetric one (e.g. ``(-3, 3)``) allows both."""

    delta_layer_id: Range | None = None
    delta_r: Range | None = None
    delta_x: Range | None = None
    delta_phi: Range | None = None


Prescription = Union[FullyConnected, Regional, ConnectionRules]


def _range(value: Any) -> Range | None:
    if value is None:
        return None
    lo, hi = value
    return (float(lo), float(hi))


def parse_prescription(spec: dict[str, Any]) -> Prescription:
    kind = spec["kind"]
    if kind == "fully_connected":
        return FullyConnected(directed=bool(spec.get("directed", False)))
    if kind == "regional":
        return Regional(phi_width=float(spec["phi_width"]))
    if kind == "connection_rules":
        return ConnectionRules(
            delta_layer_id=_range(spec.get("delta_layer_id")),
            delta_r=_range(spec.get("delta_r")),
            delta_x=_range(spec.get("delta_x")),
            delta_phi=_range(spec.get("delta_phi")),
        )
    raise ValueError(f"Unknown prescription kind: {kind!r}")
