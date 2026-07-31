"""Polygonal (octagonal) tracking stations -- the muon system's geometry.

A muon spectrometer isn't built as a cylinder: it's a small number of large
flat chambers arranged as a polygon around the beam line (an octagon, here).
Each *side* is a **triplet** -- three closely spaced parallel planes, so a
single crossing gives three measurements and a local direction, which is what
lets a muon station stand on its own as a track segment finder.

Three levels, smallest to largest:

- :func:`build_polygon` -- one closed polygon: ``n_sides``
  :class:`~detector2d.geometry.LineLayer` chords sharing one ``layer_id``
  (the same convention :mod:`detector2d.barrel` uses for the modules of one
  barrel layer).
- :func:`build_polygon_triplet_station` -- one *station*: ``n_planes``
  nested polygons ``gap`` apart, each its own ``layer_id``.
- :func:`build_muon_system` -- ``n_stations`` equally spaced stations.

Size is given by the **apothem**: the perpendicular distance from the origin
to the middle of a side (not to a vertex). A ray leaving the origin therefore
always crosses a polygon at a radius between ``apothem`` and
``apothem / cos(pi/n_sides)``, and crosses each plane exactly once -- except
aimed exactly at a *vertex*, where it clips the two sides meeting there and
that plane reports two hits. That is the polygon's own version of the module
overlap :mod:`detector2d.barrel` builds into a barrel layer, and it is real
(chambers do overlap at a station's corners), not an artifact.
"""

from __future__ import annotations

import math

from .geometry import LineLayer

TWO_PI = 2.0 * math.pi


def polygon_vertices(
    apothem: float, n_sides: int, phi_offset: float = 0.0, center: tuple[float, float] = (0.0, 0.0)
) -> list[tuple[float, float]]:
    """The ``n_sides`` vertices of a regular polygon with the given
    ``apothem``, oriented so that one side is centered exactly on azimuth
    ``phi_offset`` (side midpoints then land on every multiple of the side
    spacing ``2*pi/n_sides`` from there).

    The circumradius is ``apothem / cos(pi/n_sides)``; vertices sit halfway
    (in angle) between consecutive side midpoints -- which is why the first
    vertex is at ``phi_offset + 0.5 * spacing``.
    """
    if n_sides < 3:
        raise ValueError(f"n_sides must be >= 3, got {n_sides!r}")
    if apothem <= 0.0:
        raise ValueError(f"apothem must be > 0, got {apothem!r}")
    circumradius = apothem / math.cos(math.pi / n_sides)
    cx, cy = center
    step = TWO_PI / n_sides
    return [
        (cx + circumradius * math.cos(phi_offset + (i + 0.5) * step),
         cy + circumradius * math.sin(phi_offset + (i + 0.5) * step))
        for i in range(n_sides)
    ]


def build_polygon(
    layer_id: int,
    apothem: float,
    n_sides: int = 8,
    phi_offset: float = 0.0,
    pitch: float | None = None,
    system: str = "muon",
    center: tuple[float, float] = (0.0, 0.0),
) -> list[LineLayer]:
    """One closed polygon as ``n_sides`` ``LineLayer`` chords, all tagged with
    ``layer_id`` -- they are the sides of a single physical plane, exactly as
    a barrel layer's modules all share one ``layer_id``."""
    vertices = polygon_vertices(apothem, n_sides, phi_offset, center)
    return [
        LineLayer(
            layer_id=layer_id,
            p1=vertices[i],
            p2=vertices[(i + 1) % n_sides],
            pitch=pitch,
            system=system,
        )
        for i in range(n_sides)
    ]


def build_polygon_triplet_station(
    layer_id_base: int,
    apothem: float,
    gap: float,
    n_planes: int = 3,
    n_sides: int = 8,
    phi_offset: float = 0.0,
    pitch: float | None = None,
    system: str = "muon",
    center: tuple[float, float] = (0.0, 0.0),
) -> list[LineLayer]:
    """One station: ``n_planes`` nested polygons at apothems ``apothem``,
    ``apothem + gap``, ... Each plane gets its own ``layer_id``
    (``layer_id_base + plane``), so a track crossing the station produces one
    hit per plane -- the triplet."""
    if n_planes < 1:
        raise ValueError(f"n_planes must be >= 1, got {n_planes!r}")
    layers: list[LineLayer] = []
    for plane in range(n_planes):
        layers.extend(
            build_polygon(
                layer_id=layer_id_base + plane,
                apothem=apothem + plane * gap,
                n_sides=n_sides,
                phi_offset=phi_offset,
                pitch=pitch,
                system=system,
                center=center,
            )
        )
    return layers


def build_muon_system(
    layer_id_base: int,
    apothem_inner: float,
    station_spacing: float,
    n_stations: int = 3,
    n_planes: int = 3,
    n_sides: int = 8,
    triplet_gap: float = 8.0,
    phi_offset: float = 0.0,
    pitch: float | None = None,
    system: str = "muon",
    station_id_step: int = 10,
    center: tuple[float, float] = (0.0, 0.0),
) -> list[LineLayer]:
    """``n_stations`` equally spaced triplet stations (see
    :func:`build_polygon_triplet_station`), the innermost at
    ``apothem_inner`` and each next one ``station_spacing`` further out.

    ``layer_id``\\ s are allocated ``layer_id_base + station*station_id_step +
    plane``, leaving room between stations so the station a hit belongs to is
    readable straight off its ``layer_id``.
    """
    if n_stations < 1:
        raise ValueError(f"n_stations must be >= 1, got {n_stations!r}")
    if station_id_step <= n_planes:
        raise ValueError(
            f"station_id_step ({station_id_step}) must exceed n_planes ({n_planes}), "
            "otherwise two stations' planes would collide on the same layer_id"
        )
    layers: list[LineLayer] = []
    for station in range(n_stations):
        layers.extend(
            build_polygon_triplet_station(
                layer_id_base=layer_id_base + station * station_id_step,
                apothem=apothem_inner + station * station_spacing,
                gap=triplet_gap,
                n_planes=n_planes,
                n_sides=n_sides,
                phi_offset=phi_offset,
                pitch=pitch,
                system=system,
                center=center,
            )
        )
    return layers
