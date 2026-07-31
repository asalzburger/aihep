"""Calorimeter sampling layers: a circular layer that additionally knows its
own azimuthal *cell* structure.

A tracking layer only needs to say "a particle crossed here"; a calorimeter
layer has to say "and it put this much energy into cell 37", so the layer
itself owns the phi binning. :class:`CaloRing` is therefore a
:class:`~detector2d.geometry.CircleLayer` *subclass* -- every intersection
routine in :mod:`detector2d.intersect` keeps working on it unchanged -- with
``n_phi`` cells, an optional ``phi_offset``, and a radial ``thickness`` (used
for drawing the layer as an annulus rather than a bare circle; the
intersection math still treats it as the infinitely thin circle at
``radius``).

``phi_offset`` is what lets a stack of rings be *staggered*: shifting one
layer by half a cell means its cell boundaries fall in the middle of the
neighbouring layers' cells, so a shower landing exactly on a boundary in one
layer is still cleanly measured by the next. :func:`build_calo_stack` takes
that stagger per layer, in units of a cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import CircleLayer

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class CaloRing(CircleLayer):
    """A circular calorimeter sampling layer segmented into ``n_phi`` cells.

    Cell ``i`` spans ``[phi_offset + i*dphi, phi_offset + (i+1)*dphi)`` where
    ``dphi = 2*pi/n_phi``, measured around the ring's own center. Cell
    indices wrap, so cell ``0`` and cell ``n_phi - 1`` are neighbours.

    Inherited ``pitch`` (the arc-length cell size the clustering packages
    digitize with) is derived from ``n_phi``, not given independently -- pass
    it and it is overwritten.
    """

    n_phi: int = 1
    phi_offset: float = 0.0
    thickness: float = 0.0
    system: str = "ecal"

    def __post_init__(self) -> None:
        if self.n_phi < 1:
            raise ValueError(f"n_phi must be >= 1, got {self.n_phi!r}")
        # frozen dataclass: derive `pitch` from the cell count via object.__setattr__
        object.__setattr__(self, "pitch", TWO_PI * self.radius / self.n_phi)

    @property
    def dphi(self) -> float:
        """Angular width of one cell."""
        return TWO_PI / self.n_phi

    def cell_index(self, phi: float) -> int:
        """Index of the cell containing azimuth ``phi`` (radians, any branch)."""
        return int(math.floor((phi - self.phi_offset) % TWO_PI / self.dphi)) % self.n_phi

    def cell_edges(self, index: int) -> tuple[float, float]:
        """``(phi_low, phi_high)`` of cell ``index``, unwrapped (``phi_high >
        phi_low``) and anchored at ``phi_offset``, so the two edges of one
        cell are always directly comparable even across the +-pi branch cut."""
        low = self.phi_offset + (index % self.n_phi) * self.dphi
        return (low, low + self.dphi)

    def cell_center_phi(self, index: int) -> float:
        """Azimuth of the middle of cell ``index``, wrapped into (-pi, pi]."""
        low, high = self.cell_edges(index)
        return (0.5 * (low + high) + math.pi) % TWO_PI - math.pi

    def cell_position(self, index: int) -> tuple[float, float]:
        """``(x, y)`` of cell ``index``'s center, on the ring itself."""
        phi = self.cell_center_phi(index)
        cx, cy = self.center
        return (cx + self.radius * math.cos(phi), cy + self.radius * math.sin(phi))

    def cell_local_coord(self, index: int) -> float:
        """Cell center as a ``local_coord`` on the ring -- arc length from the
        +x axis, the same convention :mod:`detector2d.intersect` reports for a
        :class:`~detector2d.geometry.CircleLayer` hit."""
        return self.radius * self.cell_center_phi(index)


@dataclass
class CaloStackConfig:
    """Declarative description of one calorimeter (a radial stack of rings)."""

    layer_id_base: int
    r_inner: float
    n_layers: int
    thickness: float
    n_phi: int
    system: str = "ecal"
    #: Per-layer azimuthal shift, in units of one cell. Padded with 0.0 (and
    #: truncated) to ``n_layers``, so ``[0.0, 0.5]`` on a 3-layer stack means
    #: "shift the middle layer by half a cell, leave the others aligned".
    phi_stagger: list[float] = field(default_factory=list)


def build_calo_stack(
    layer_id_base: int,
    r_inner: float,
    n_layers: int,
    thickness: float,
    n_phi: int,
    system: str = "ecal",
    phi_stagger: list[float] | tuple[float, ...] = (),
    center: tuple[float, float] = (0.0, 0.0),
) -> list[CaloRing]:
    """A radial stack of ``n_layers`` :class:`CaloRing`\\ s, each of radial
    ``thickness``, filling outward from ``r_inner``.

    Ring ``i`` sits at the *middle* of its own slab -- ``r_inner + (i + 0.5) *
    thickness`` -- which is where a sampling layer's measurement effectively
    is, and which keeps the stack's full radial extent exactly
    ``[r_inner, r_inner + n_layers*thickness]``. ``layer_id`` runs
    ``layer_id_base, layer_id_base + 1, ...``.

    ``phi_stagger[i]`` shifts ring ``i`` by that fraction of a cell (see the
    module docstring); missing entries are 0.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers!r}")
    stagger = list(phi_stagger) + [0.0] * max(0, n_layers - len(phi_stagger))
    rings = []
    for i in range(n_layers):
        radius = r_inner + (i + 0.5) * thickness
        rings.append(
            CaloRing(
                layer_id=layer_id_base + i,
                center=center,
                radius=radius,
                n_phi=n_phi,
                phi_offset=stagger[i] * TWO_PI / n_phi,
                thickness=thickness,
                system=system,
            )
        )
    return rings


def build_calo_stack_from_config(config: CaloStackConfig) -> list[CaloRing]:
    return build_calo_stack(
        layer_id_base=config.layer_id_base,
        r_inner=config.r_inner,
        n_layers=config.n_layers,
        thickness=config.thickness,
        n_phi=config.n_phi,
        system=config.system,
        phi_stagger=config.phi_stagger,
    )
